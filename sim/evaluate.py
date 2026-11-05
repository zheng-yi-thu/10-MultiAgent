"""在测试日上对比各方法，并做扩容、通信中断、控制器故障测试。结果保存为 results.npz 与 results.txt。"""
import time
import numpy as np
import torch
from env import Config, ClusterEnv, wind_profiles, T_STEPS, DT
from baselines import rule_policy, milp_schedule
from train import Actor

TEST_DAYS, TEST_SEED = 20, 123


def run(cfg, wind, policy, hook=None):
    env = ClusterEnv(cfg, wind)
    o = env.reset()
    rec = {k: [] for k in ["h2", "p_ess", "curtail", "shed", "starts", "revenue", "soc", "p_el"]}
    X, dts = [], []
    for t in range(T_STEPS):
        if hook:
            hook(env, t)
        t0 = time.perf_counter()
        x = policy(env, o, t)
        dts.append(time.perf_counter() - t0)
        if hook and hasattr(env, "dead"):
            x = x.copy(); x[:, env.dead] = 0.0
        o, _, _, info = env.step(x)
        X.append(env.x.copy())
        for k in rec:
            rec[k].append(info[k])
    rec = {k: np.array(v) for k, v in rec.items()}
    rec["x"] = np.array(X)
    rec["step_time"] = np.mean(dts)
    return rec


def summary(rec):
    return dict(
        revenue=rec["revenue"].sum(0).mean(),
        h2=rec["h2"].sum(0).mean(),
        starts=rec["starts"].sum(0).mean(),
        ess_thru=(np.abs(rec["p_ess"]) * DT).sum(0).mean(),
        curtail=(rec["curtail"] * DT).sum(0).mean(),
        shed=(rec["shed"] * DT).sum(0).mean(),
        soc_min=rec["soc"].min(),
        soc_max=rec["soc"].max(),
    )


def rl_policy(actor):
    def f(env, o, t):
        with torch.no_grad():
            return actor(torch.from_numpy(o)).numpy()   # 部署时取确定性输出
    return f


def milp_policy(cfg, wind):
    sched, times = [], []
    for d in range(wind.shape[0]):
        x, dt = milp_schedule(cfg, wind[d], 120)
        sched.append(x); times.append(dt)
    sched = np.stack(sched, 1)  # (T,E,N)
    return (lambda env, o, t: sched[t]), np.mean(times)


def main():
    cfg = Config()
    # 在验证日（与测试日不重合）上挑选训练效果最好的随机种子
    val_wind = wind_profiles(10, cfg.p_wind_cap, 777)
    best = None
    for s in range(3):
        a = Actor()
        a.load_state_dict(torch.load(f"actor_s{s}.pt"))
        v = summary(run(cfg, val_wind, rl_policy(a)))["revenue"]
        print(f"种子 {s} 验证日收益 {v:.1f}")
        if best is None or v > best[0]:
            best = (v, s, a)
    actor = best[2]
    torch.save(actor.state_dict(), "actor.pt")
    print(f"选用种子 {best[1]}")
    wind = wind_profiles(TEST_DAYS, cfg.p_wind_cap, TEST_SEED)
    res, lines = {}, []
    mp, milp_t = milp_policy(cfg, wind)
    methods = {
        "均分规则": lambda env, o, t: rule_policy(env, "equal"),
        "效率优先规则": lambda env, o, t: rule_policy(env, "eff"),
        "完美预测混合整数规划": mp,
        "本发明": rl_policy(actor),
    }
    for name, pol in methods.items():
        r = run(cfg, wind, pol)
        res[name] = r
        s = summary(r)
        lines.append(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in s.items()) + f", step_ms={r['step_time']*1000:.3f}")
    lines.append(f"混合整数规划每天求解耗时 {milp_t:.2f} s")

    # 扩容：10 台电解槽直接复制同一个策略网络
    cfg10 = Config(n_el=10)
    for name, pol in {"均分规则": lambda env, o, t: rule_policy(env, "equal"),
                      "效率优先规则": lambda env, o, t: rule_policy(env, "eff"),
                      "本发明": rl_policy(actor)}.items():
        lines.append(f"[10台] {name}: " + ", ".join(f"{k}={v:.3f}" for k, v in summary(run(cfg10, wind, pol)).items()))

    # 通信中断：8 至 14 时，第 3、4、5 台收不到储能信息
    def link_hook(env, t):
        env.link_ok[:] = True
        if 96 <= t < 168:
            env.link_ok[:, 2:5] = False
    r = run(cfg, wind, rl_policy(actor), link_hook)
    res["通信中断"] = r
    lines.append("[通信中断] 本发明: " + ", ".join(f"{k}={v:.3f}" for k, v in summary(r).items()))

    # 控制器故障：第 2 台全天停运，其余照常
    def dead_hook(env, t):
        env.dead = 1
    r = run(cfg, wind, rl_policy(actor), dead_hook)
    lines.append("[控制器故障] 本发明: " + ", ".join(f"{k}={v:.3f}" for k, v in summary(r).items()))

    open("results.txt", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
    np.savez("results.npz", wind=wind, **{f"{k}__{f}": v[f] for k, v in res.items() for f in ["x", "p_ess", "soc", "p_el", "h2", "starts"]})


if __name__ == "__main__":
    main()
