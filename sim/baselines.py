"""对比方法：两种集中式规则调度，以及已知全天风电的集中式混合整数规划（理论上界）。"""
import time
import numpy as np
import highspy
from env import DT, T_STEPS, h2_rate


def rule_policy(env, mode):
    """集中式规则：中央控制器测得风电功率，按储能电量修正后得到集群目标功率，再分给各台电解槽。"""
    c, N = env.cfg, env.cfg.n_el
    p_wind = env.wind[:, env.t]
    target = p_wind + 2.0 * (env.soc - 0.5) * c.ess_p
    target = np.clip(target, 0, N * c.p_el_rated)
    if mode == "equal":      # 能开的都开，平均分配
        n_on = np.where(target / N >= c.x_min, N, np.floor(target / c.x_min))
    else:                    # 效率优先：按最优负荷率估算台数，少开几台；带回差，避免频繁启停
        n_req = np.ceil(target / (c.x_opt.mean() * c.p_el_rated))
        n_prev = (env.x > 0).sum(1)
        up = target > 0.95 * n_prev * c.p_el_rated
        down = target < 0.8 * np.maximum(n_prev - 1, 0) * c.x_opt.mean() * c.p_el_rated
        n_on = np.where(up | down, n_req, n_prev)
    n_on = np.clip(n_on, 0, N)
    share = np.where(n_on > 0, target / np.maximum(n_on, 1), 0)
    order = np.argsort(env.rank, 1)
    x = np.zeros((env.E, N))
    for e in range(env.E):
        k = int(n_on[e])
        x[e, order[e, :k]] = share[e] / c.p_el_rated
    return x


def milp_schedule(cfg, wind_day, time_limit=120):
    """已知全天风电的集中式日前调度（忽略温度影响），返回 (负荷率序列 T×N, 求解耗时 s)。"""
    N, T = cfg.n_el, T_STEPS
    # 产氢速率对功率的分段线性上包络（按 80℃ 计算）
    xs = np.linspace(cfg.x_min, 1.0, 6)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", 0.005)
    inf = highspy.kHighsInf
    idx = {}

    def var(name, lb, ub, cost, integer=False):
        h.addVar(lb, ub)
        j = h.getNumCol() - 1
        h.changeColCost(j, cost)
        if integer:
            h.changeColIntegrality(j, highspy.HighsVarType.kInteger)
        idx[name] = j
        return j

    # 目标为最小化（负收益）
    for t in range(T):
        for i in range(N):
            var(("u", t, i), 0, 1, 0, True)
            var(("s", t, i), 0, 1, cfg.cost_start)
            var(("p", t, i), 0, cfg.p_el_rated, 0)
            var(("h", t, i), 0, inf, -cfg.price_h2 * DT)
        var(("c", t), 0, cfg.ess_p, cfg.cost_ess * DT)
        var(("d", t), 0, cfg.ess_p, cfg.cost_ess * DT)
        var(("w", t), 0, inf, 0)          # 弃风
        var(("E", t), cfg.soc_hard[0] * cfg.ess_e, cfg.soc_hard[1] * cfg.ess_e, 0)

    def row(lb, ub, terms):
        h.addRow(lb, ub, len(terms), np.array([idx[k] for k, _ in terms], np.int32), np.array([v for _, v in terms], float))

    hv_all = h2_rate(cfg, np.repeat(xs[:, None], N, 1), np.full((len(xs), N), 80.0))  # (6, N)
    for t in range(T):
        for i in range(N):
            u, s, p, hh = ("u", t, i), ("s", t, i), ("p", t, i), ("h", t, i)
            row(-inf, 0, [(p, 1), (u, -cfg.p_el_rated)])
            row(0, inf, [(p, 1), (u, -cfg.x_min * cfg.p_el_rated)])
            for k in range(len(xs) - 1):
                a = (hv_all[k + 1, i] - hv_all[k, i]) / (xs[k + 1] - xs[k])
                b = hv_all[k, i] - a * xs[k]
                row(-inf, 0, [(hh, 1), (p, -a), (u, -b)])
            prev_u = ("u", t - 1, i)
            if t == 0:
                row(0, inf, [(s, 1), (u, -1)])
                row(-inf, cfg.ramp, [(p, 1)])
            else:
                row(0, inf, [(s, 1), (u, -1), (prev_u, 1)])
                row(-inf, cfg.ramp, [(p, 1), (("p", t - 1, i), -1)])
                # 降负荷受爬坡限制，停机时允许一步降到 0
                row(-inf, cfg.ramp + 1.0, [(("p", t - 1, i), 1), (p, -1), (u, 1.0)])
        # 功率平衡：风电 - 弃风 = 电解槽 + 热备 + 充电 - 放电
        terms = [(("p", t, i), 1) for i in range(N)] + [(("u", t, i), -cfg.p_standby) for i in range(N)]
        terms += [(("c", t), 1), (("d", t), -1), (("w", t), 1)]
        rhs = wind_day[t] - N * cfg.p_standby
        row(rhs, rhs, terms)
        e_prev = 0.5 * cfg.ess_e if t == 0 else None
        terms = [(("E", t), 1), (("c", t), -cfg.eta * DT), (("d", t), DT / cfg.eta)]
        if t == 0:
            row(e_prev, e_prev, terms)
        else:
            terms.append((("E", t - 1), -1))
            row(0, 0, terms)
    row(0.5 * cfg.ess_e, inf, [(("E", T - 1), 1)])
    t0 = time.perf_counter()
    h.run()
    dt = time.perf_counter() - t0
    sol = np.array(h.getSolution().col_value)
    x = np.array([[sol[idx[("p", t, i)]] for i in range(N)] for t in range(T)]) / cfg.p_el_rated
    u = np.array([[sol[idx[("u", t, i)]] for i in range(N)] for t in range(T)])
    return np.where(u > 0.5, x, 0.0), dt
