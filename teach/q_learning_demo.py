"""强化学习入门算例：一台电解槽 + 一块储能，用 Q 学习学会“什么时候开、开多大”。

运行：python q_learning_demo.py
全部用 numpy，几秒钟跑完，会打印学到的策略表并画出学习曲线。

======================== 问题设定 ========================
每一步代表 1 小时，一个回合（episode）= 一天 = 24 步。
风电只有三档出力：0、1、2 MW，每小时按一定概率变化。
储能容量 4 MWh，电量只取 0、1、2、3、4 五个整数值。
电解槽有三种动作：停机（0 MW）、低负荷（1 MW）、满负荷（2 MW）。
风电多出来的电充进储能，不够的由储能补；储能满了就弃风，储能空了就只能强制停机。

======================== 基本概念 ========================
状态 s    智能体做决定时能看到的信息。这里是 (风电档位, 储能电量, 电解槽是否在运行)。
动作 a    智能体能做的选择。这里是 停机 / 低负荷 / 满负荷。
奖励 r    每一步结束后环境给的分数。这里 = 产氢收益 - 启动成本 - 强制停机惩罚。
策略 π    从状态到动作的规则，即“看到什么就做什么”。学习的目的就是找到好的策略。
Q 值      Q(s, a) 表示“在状态 s 做动作 a，之后一直按好策略走，能拿到的总分”。
          有了准确的 Q 值，策略就很简单：每个状态选 Q 值最大的动作。
折扣 γ    未来的奖励打折计入，γ=0.95 表示一小时后的 1 分只算 0.95 分。
          γ 越大，智能体越看重长远，比如为了晚上的风电少而提前给储能留电。
探索 ε    学习时以 ε 的概率随便试一个动作，否则就只会重复已知的做法，发现不了更好的。
"""
import numpy as np

rng = np.random.default_rng(0)

# ---------------- 环境参数 ----------------
WIND_MW = [0, 1, 2]            # 风电三档出力
SOC_MAX = 4                    # 储能最大电量 MWh
ACTIONS = ["停机", "低负荷", "满负荷"]
EL_MW = [0, 1, 2]              # 三种动作对应的电解槽功率
H2_KG = [0, 15, 40]            # 每小时产氢 kg。低负荷时气体互串、效率低，所以 1 MW 只产 15 kg
PRICE = 20                     # 氢气 元/kg
START_COST = 100               # 每次从停机到运行的衰减成本 元
SHED_COST = 500                # 储能放空导致强制停机的惩罚 元
ESS_COST = 40                  # 储能每充放 1 MWh 的衰减成本 元。去掉这一项，智能体会学会“先停机把电存起来”的钻空子做法
STEPS = 24


def wind_next(w):
    """风电档位的变化：70% 保持不变，其余概率变到相邻档位。"""
    p = rng.random()
    if p < 0.15:
        return max(w - 1, 0)
    if p < 0.30:
        return min(w + 1, 2)
    return w


def step(state, a):
    """环境：给定当前状态和动作，返回下一状态和奖励。智能体不知道这个函数的内部，只能通过试来学习。"""
    w, soc, running = state
    p_el = EL_MW[a]
    reward = 0.0
    # 电量收支：风电 - 电解槽，正数充进储能，负数由储能补
    net = WIND_MW[w] - p_el
    soc_new = soc + net
    if soc_new < 0:            # 储能不够，电解槽被迫停机
        reward -= SHED_COST
        a, p_el = 0, 0
        soc_new = min(soc + WIND_MW[w], SOC_MAX)
    soc_new = min(soc_new, SOC_MAX)   # 超出部分弃风
    reward -= ESS_COST * abs(soc_new - soc)   # 储能每充放 1 MWh 都有电池衰减成本
    if a > 0 and not running:
        reward -= START_COST
    reward += PRICE * H2_KG[a]
    return (wind_next(w), soc_new, int(a > 0)), reward


def idx(state):
    """把状态编成表格的行号：3 档风 × 5 档电量 × 2 种运行状态 = 30 行。"""
    w, soc, running = state
    return (w * (SOC_MAX + 1) + soc) * 2 + running


N_STATES = 3 * (SOC_MAX + 1) * 2


def run_episode(policy):
    """按给定策略跑一天，返回总奖励。"""
    s, total = (1, 2, 0), 0.0
    for _ in range(STEPS):
        s, r = step(s, policy(s))
        total += r
    return total


# ---------------- Q 学习 ----------------
def q_learning(episodes=200000, gamma=0.95):
    Q = np.zeros((N_STATES, 3))           # Q 表：每行一个状态，每列一个动作，初始全为 0
    visits = np.zeros((N_STATES, 3))      # 记录每个格子被试过几次。没试过的格子 Q 值一直是 0，学不到东西
    curve = []
    for ep in range(episodes):
        eps = max(0.02, 1.0 - ep / (0.6 * episodes))   # 探索率从 1 逐渐降到 0.02：先多试，后多用
        s = (rng.integers(3), rng.integers(SOC_MAX + 1), rng.integers(2))   # 每天从随机状态开始，让各种情况都有机会遇到
        total = 0.0
        for t in range(STEPS):
            # 选动作：以 ε 的概率随便试，否则选当前 Q 值最大的
            a = rng.integers(3) if rng.random() < eps else int(np.argmax(Q[idx(s)]))
            s2, r = step(s, a)
            total += r
            # 核心更新：用“这一步的奖励 + 下一状态能拿到的最好 Q 值”来修正当前 Q 值
            # 日末不清零，储能里剩的电第二天还能用，所以每一天都接着往后算
            target = r + gamma * Q[idx(s2)].max()
            visits[idx(s)][a] += 1
            alpha = max(0.005, 1.0 / visits[idx(s)][a] ** 0.6)   # 学习率：刚开始大步修正，试得越多步子越小，把随机风电的噪声平均掉
            Q[idx(s)][a] += alpha * (target - Q[idx(s)][a])
            s = s2
        curve.append(total)
    return Q, np.array(curve), visits


if __name__ == "__main__":
    Q, curve, visits = q_learning()
    print(f"最少被试过的格子只试了 {int(visits.min())} 次，最多的 {int(visits.max())} 次\n")

    learned = lambda s: int(np.argmax(Q[idx(s)]))
    follow = lambda s: s[0]                         # 规则：风电几档就开几档
    random_ = lambda s: int(rng.integers(3))

    print("=== 三种策略各跑 2000 天的平均日收益（元） ===")
    for name, pol in [("随机乱选", random_), ("跟着风电走", follow), ("Q 学习", learned)]:
        print(f"{name:8s} {np.mean([run_episode(pol) for _ in range(2000)]):8.0f}")

    print("\n=== 学到的策略表（电解槽当前在运行时） ===")
    print("储能电量 →   " + "  ".join(f"{k} MWh " for k in range(SOC_MAX + 1)))
    for w in range(3):
        row = [ACTIONS[learned((w, soc, 1))] for soc in range(SOC_MAX + 1)]
        print(f"风电 {WIND_MW[w]} MW    " + "  ".join(f"{a:4s}" for a in row))
    print("\n=== 学到的策略表（电解槽当前停机时） ===")
    for w in range(3):
        row = [ACTIONS[learned((w, soc, 0))] for soc in range(SOC_MAX + 1)]
        print(f"风电 {WIND_MW[w]} MW    " + "  ".join(f"{a:4s}" for a in row))

    print("\n=== 一个状态的 Q 值：风电 1 MW、储能 2 MWh、正在运行 ===")
    for a in range(3):
        print(f"{ACTIONS[a]:4s} Q = {Q[idx((1, 2, 1))][a]:8.0f}")

    try:
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
        plt.rcParams["axes.unicode_minus"] = False
        smooth = np.convolve(curve, np.ones(2000) / 2000, mode="valid")
        plt.figure(figsize=(6, 3.5))
        plt.plot(smooth, color="k")
        plt.xlabel("训练回合（天）")
        plt.ylabel("日收益（元，2000 天滑动平均）")
        plt.title("学习曲线：前期探索多、收益低，后期收敛")
        plt.tight_layout()
        plt.savefig("learning_curve.png", dpi=150)
        print("\n学习曲线已保存到 learning_curve.png")
    except ImportError:
        pass
