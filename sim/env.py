"""离网风电-储能构网-碱性电解槽集群仿真环境（向量化，可同时仿真多天）。

时间步长 5 分钟，一天 288 步。储能负责维持母线，吸收风电与电解槽之间的功率差额。
"""
import numpy as np

DT = 5 / 60          # 步长 (h)
T_STEPS = 288        # 一天的步数


class Config:
    def __init__(self, n_el=8, seed=0):
        self.n_el = n_el
        self.p_wind_cap = 12.0      # 风电装机 (MW)
        self.p_el_rated = 1.0       # 单台电解槽额定功率 (MW)
        self.x_min = 0.3            # 最低负荷率（气体纯度限制）
        self.ramp = 0.3             # 每步最大负荷率变化
        self.p_standby = 0.02       # 热备功耗 (MW)
        self.ess_p = 2.0            # 储能额定功率 (MW)
        self.ess_e = 4.0            # 储能容量 (MWh)
        self.eta = 0.95             # 储能单向效率
        self.soc_hard = (0.05, 0.95)
        self.soc_safe = (0.15, 0.85)  # 安全规则触发阈值
        # 经济参数（元）
        self.price_h2 = 2.0         # 氢气价值 元/Nm3
        self.cost_ess = 100.0       # 储能吞吐成本 元/MWh
        self.cost_start = 150.0     # 每次启动的衰减成本 元
        self.cost_shed = 20000.0    # 失负荷惩罚 元/MWh（母线失稳风险）
        rng = np.random.default_rng(1000 + seed)
        # 各台电解槽效率参数存在制造与老化差异
        self.x_opt = rng.uniform(0.6, 0.8, n_el)
        self.sec_min = rng.uniform(4.3, 4.6, n_el)   # kWh/Nm3


def sec(cfg, x, temp):
    """比能耗 kWh/Nm3，随负荷率和温度变化。"""
    x = np.maximum(x, 1e-3)
    base = cfg.sec_min * (1 + 0.25 * (x - cfg.x_opt) ** 2)
    temp_f = 1 + 0.003 * np.maximum(80.0 - temp, 0.0)   # 低温时效率下降
    faraday = x / (x + 0.02)                            # 低负荷时气体互串导致效率下降
    return base * temp_f / faraday


def h2_rate(cfg, x, temp):
    """产氢速率 Nm3/h。"""
    return np.where(x > 0, 1000 * x * cfg.p_el_rated / sec(cfg, x, temp), 0.0)


def wind_profiles(n_days, cap, seed):
    """风速采用均值回复随机过程，经风机功率曲线换算为风电出力 (MW)，形状 (n_days, T_STEPS)。"""
    rng = np.random.default_rng(seed)
    t = np.arange(T_STEPS) * DT
    out = np.zeros((n_days, T_STEPS))
    for d in range(n_days):
        mean_v = rng.uniform(5.5, 10.0)
        phase = rng.uniform(0, 24)
        mu = mean_v + 1.8 * np.sin(2 * np.pi * (t - phase) / 24)
        v = np.zeros(T_STEPS)
        v[0] = mu[0] + rng.normal(0, 1.5)
        for k in range(1, T_STEPS):
            v[k] = v[k - 1] + 0.08 * (mu[k] - v[k - 1]) + rng.normal(0, 0.45)
        v = np.maximum(v, 0)
        p = np.clip((v - 3.0) / (11.0 - 3.0), 0, 1) ** 3
        p[v >= 25] = 0
        out[d] = cap * p
    return out


class ClusterEnv:
    """E 天并行仿真。每台电解槽的观测只包含本机信息和沿环形通信链路转发来的储能信息。"""

    OBS_DIM = 7

    def __init__(self, cfg, wind):
        self.cfg = cfg
        self.wind = wind
        self.E = wind.shape[0]

    def reset(self, day_offset=0):
        c, E, N = self.cfg, self.E, self.cfg.n_el
        self.t = 0
        self.x = np.zeros((E, N))
        self.temp = np.full((E, N), 60.0)
        self.soc = np.full(E, 0.5)
        self.p_ess = np.zeros(E)          # 储能功率，正为充电
        # 轮值序号每天轮换，使各台电解槽的运行时长保持均衡
        self.rank = ((np.arange(N)[None, :] + day_offset + np.arange(E)[:, None]) % N) / max(N - 1, 1)
        self.ess_info = np.stack([self.soc, self.p_ess, self.p_ess], 1)[:, None, :].repeat(N, 1)  # 各台收到的储能信息
        self.link_ok = np.ones((E, N), bool)  # False 表示该台收不到新的储能信息
        return self.obs()

    def obs(self):
        o = np.stack([
            (self.temp - 25) / 55,
            self.x,
            (self.x > 0).astype(float),
            self.ess_info[..., 0],
            self.ess_info[..., 1] / self.cfg.ess_p,
            self.rank,
            self.ess_info[..., 2] / self.cfg.ess_p,
        ], -1)
        return o.astype(np.float32)

    def global_state(self):
        """仅训练阶段供评价网络使用的全局信息。"""
        hour = self.t * DT
        g = np.concatenate([
            self.obs().reshape(self.E, -1),
            (self.wind[:, min(self.t, T_STEPS - 1)] / self.cfg.p_wind_cap)[:, None],
            np.full((self.E, 1), np.sin(2 * np.pi * hour / 24)),
            np.full((self.E, 1), np.cos(2 * np.pi * hour / 24)),
            np.full((self.E, 1), self.t / T_STEPS),
        ], 1)
        return g.astype(np.float32)

    def safety(self, x_req, info):
        """安全规则层：最低负荷、爬坡限制、储能电量保护。只使用本机信息和收到的储能信息。"""
        c = self.cfg
        x_req = np.clip(x_req, 0, 1)
        soc, p_ess = info[..., 0], info[..., 1]
        # 储能电量过低且仍在放电，或放电功率接近额定值，强制降负荷；充电侧同理强制升负荷
        low = ((soc < c.soc_safe[0]) & (p_ess < 0)) | (p_ess < -0.8 * c.ess_p)
        high = ((soc > c.soc_safe[1]) & (p_ess > 0)) | (p_ess > 0.8 * c.ess_p)
        run = self.x > 0   # 强制调节只作用于正在运行的电解槽，不强制启动热备电解槽
        x_req = np.where(low & run, np.minimum(x_req, self.x - c.ramp), x_req)
        x_req = np.where(high & run, np.maximum(x_req, self.x + c.ramp), x_req)
        x_new = np.clip(x_req, self.x - c.ramp, self.x + c.ramp)
        x_new = np.clip(x_new, 0, 1)
        # 低于最低负荷则转入热备
        x_new = np.where(x_new < c.x_min, np.where(self.x > 0, 0.0, np.where(x_req >= c.x_min, c.x_min, 0.0)), x_new)
        return x_new

    def step(self, x_req):
        c, E = self.cfg, self.E
        x_new = self.safety(x_req, self.ess_info)
        starts = ((x_new > 0) & (self.x == 0)).sum(1)
        p_wind = self.wind[:, self.t]
        p_el = (x_new * c.p_el_rated).sum(1) + c.p_standby * (x_new == 0).sum(1)
        # 储能吸收差额，受额定功率与电量限制
        p_need = p_wind - p_el
        e = self.soc * c.ess_e
        ch_max = np.minimum(c.ess_p, (c.soc_hard[1] * c.ess_e - e) / (c.eta * DT))
        dis_max = np.minimum(c.ess_p, (e - c.soc_hard[0] * c.ess_e) * c.eta / DT)
        p_ess = np.clip(p_need, -np.maximum(dis_max, 0), np.maximum(ch_max, 0))
        curtail = np.maximum(p_need - p_ess, 0)        # 储能充满后的弃风
        shed = np.maximum(p_ess - p_need, 0)           # 储能放空后被迫切除的电解槽负荷
        if np.any(shed > 0):
            run_p = (x_new * c.p_el_rated).sum(1)
            ratio = np.where(run_p > 0, np.clip(1 - shed / np.maximum(run_p, 1e-9), 0, 1), 1)
            x_new = x_new * ratio[:, None]
        self.soc = (e + (c.eta * np.maximum(p_ess, 0) + np.minimum(p_ess, 0) / c.eta) * DT) / c.ess_e
        # 温度：运行时趋向与负荷相关的工作温度，热备时缓慢下降
        t_target = np.where(x_new > 0, 55 + 25 * np.minimum(x_new / 0.7, 1), 25.0)
        tau = np.where(x_new > 0, 1.0, 4.0)
        self.temp = self.temp + (t_target - self.temp) * DT / tau
        h2 = h2_rate(c, x_new, self.temp).sum(1) * DT
        revenue = c.price_h2 * h2 - c.cost_ess * np.abs(p_ess) * DT - c.cost_start * starts - c.cost_shed * shed * DT
        soc_pen = 1000.0 * (np.maximum(c.soc_safe[0] + 0.05 - self.soc, 0) + np.maximum(self.soc - c.soc_safe[1] + 0.05, 0))
        p_ess_old = self.p_ess
        self.x, self.p_ess = x_new, p_ess
        # 储能信息沿环转发，链路中断的电解槽保持上一次收到的值
        fresh = np.stack([self.soc, self.p_ess, self.p_ess - p_ess_old], 1)[:, None, :]
        self.ess_info = np.where(self.link_ok[..., None], fresh, self.ess_info)
        self.t += 1
        done = self.t >= T_STEPS
        if done:  # 日末电量低于初始值的部分按缺额计价，避免透支储能
            revenue = revenue - 1.2 * c.price_h2 / 4.5 * 1000 * np.maximum(0.5 - self.soc, 0) * c.ess_e
        info = dict(h2=h2, p_ess=p_ess, curtail=curtail, shed=shed, starts=starts, revenue=revenue, soc=self.soc.copy(), p_el=p_el)
        return self.obs(), (revenue - soc_pen) / 100.0, done, info
