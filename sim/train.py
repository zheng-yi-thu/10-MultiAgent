"""多智能体近端策略优化训练：各台电解槽共享同一个策略网络，评价网络仅训练时使用。"""
import time
import numpy as np
import torch
import torch.nn as nn
from env import Config, ClusterEnv, wind_profiles, T_STEPS

torch.set_num_threads(6)


class Actor(nn.Module):
    """策略网络（部署在每台电解槽本地）：输入 6 个本地观测，输出负荷率。"""

    def __init__(self, obs_dim=ClusterEnv.OBS_DIM, hid=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, 1))
        self.log_std = nn.Parameter(torch.tensor([-1.0]))

    def forward(self, o):
        return torch.sigmoid(self.net(o)).squeeze(-1) * 1.2 - 0.1   # 输出范围略宽于 [0,1]，便于取到 0 和 1

    def dist(self, o):
        return torch.distributions.Normal(self(o), self.log_std.exp())


class Critic(nn.Module):
    """评价网络（仅训练阶段）：输入全局状态，估计全队未来收益。"""

    def __init__(self, s_dim, hid=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(s_dim, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, 1))

    def forward(self, s):
        return self.net(s).squeeze(-1)


def train(iters=400, E=64, seed=0, out="actor.pt"):
    cfg = Config(seed=0)
    N = cfg.n_el
    torch.manual_seed(seed)
    actor = Actor()
    env0 = ClusterEnv(cfg, wind_profiles(E, cfg.p_wind_cap, 0))
    env0.reset()
    critic = Critic(env0.global_state().shape[1])
    opt_a = torch.optim.Adam(actor.parameters(), lr=3e-4)
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3)
    gamma, lam, clip = 0.99, 0.95, 0.2
    log = []
    t0 = time.time()
    for it in range(iters):
        for g_ in opt_a.param_groups: g_["lr"] = 3e-4 * (1 - it / iters) + 3e-5
        for g_ in opt_c.param_groups: g_["lr"] = 1e-3 * (1 - it / iters) + 1e-4
        env = ClusterEnv(cfg, wind_profiles(E, cfg.p_wind_cap, 10_000 + it))
        o = env.reset(day_offset=it)
        O, S, A, LP, R = [], [], [], [], []
        for t in range(T_STEPS):
            s = env.global_state()
            with torch.no_grad():
                d = actor.dist(torch.from_numpy(o))
                a = d.sample()
                lp = d.log_prob(a)
            o2, r, done, info = env.step(a.numpy())
            O.append(o); S.append(s); A.append(a.numpy()); LP.append(lp.numpy()); R.append(r)
            o = o2
        O, S, A, LP, R = map(np.array, (O, S, A, LP, R))   # O:(T,E,N,6) S:(T,E,ds) R:(T,E)
        with torch.no_grad():
            V = critic(torch.from_numpy(S)).numpy()
        adv = np.zeros_like(R)
        g = 0
        for t in reversed(range(T_STEPS)):
            v_next = V[t + 1] if t + 1 < T_STEPS else 0
            delta = R[t] + gamma * v_next - V[t]
            g = delta + gamma * lam * g
            adv[t] = g
        ret = adv + V
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        # 团队共享收益，每台电解槽使用同一个优势值
        o_t = torch.from_numpy(O.reshape(-1, N, O.shape[-1]))
        a_t = torch.from_numpy(A.reshape(-1, N))
        lp_t = torch.from_numpy(LP.reshape(-1, N))
        adv_t = torch.from_numpy(adv_n.reshape(-1)).float()
        s_t = torch.from_numpy(S.reshape(-1, S.shape[-1]))
        ret_t = torch.from_numpy(ret.reshape(-1)).float()
        B = o_t.shape[0]
        for _ in range(8):
            perm = torch.randperm(B)
            for k in range(0, B, 2048):
                mb = perm[k:k + 2048]
                d = actor.dist(o_t[mb])
                ratio = (d.log_prob(a_t[mb]) - lp_t[mb]).exp()
                A_ = adv_t[mb][:, None]
                loss_a = -torch.min(ratio * A_, ratio.clamp(1 - clip, 1 + clip) * A_).mean() - 0.001 * d.entropy().mean()
                opt_a.zero_grad(); loss_a.backward(); nn.utils.clip_grad_norm_(actor.parameters(), 0.5); opt_a.step()
                loss_c = ((critic(s_t[mb]) - ret_t[mb]) ** 2).mean()
                opt_c.zero_grad(); loss_c.backward(); nn.utils.clip_grad_norm_(critic.parameters(), 0.5); opt_c.step()
        ep_ret = R.sum(0).mean()
        log.append(ep_ret)
        if it % 10 == 0:
            print(f"iter {it:4d}  日收益(缩放) {ep_ret:8.2f}  std {actor.log_std.exp().item():.3f}  {time.time() - t0:6.0f}s", flush=True)
    torch.save(actor.state_dict(), out)
    np.save(out.replace(".pt", "_log.npy"), np.array(log))
    return actor


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    train(iters=int(sys.argv[1]) if len(sys.argv) > 1 else 400, seed=seed, out=f"actor_s{seed}.pt")
