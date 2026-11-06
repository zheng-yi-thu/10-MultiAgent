"""生成交底书附图，黑线白底。中文用微软雅黑，英文和数字用 Times New Roman。

用法 python make_figs.py，仿真结果读取 ../sim/runs/round4/eval_data.npz 与 eval.txt。
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon

plt.rcParams.update({
    "font.family": ["Times New Roman", "Microsoft YaHei"],   # 西文字符取 Times New Roman，中文回退到微软雅黑
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
})
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "..", "sim", "runs", "round4")
CM = 1 / 2.54
LW = 1.0


def canvas(w_cm, h_cm, xmax, ymax):
    fig = plt.figure(figsize=(w_cm * CM, h_cm * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text="", ls="-", fs=9, lw=LW):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="k", lw=lw, ls=ls))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.3)


def arrow(ax, pts, ls="-", head=True, lw=LW):
    """沿折线画连线，末端带箭头。"""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color="k", lw=lw, ls=ls, solid_capstyle="butt")
    if head:
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        L = np.hypot(x2 - x1, y2 - y1)
        ax.annotate("", xy=pts[-1], xytext=(x2 - (x2 - x1) / L * 0.8, y2 - (y2 - y1) / L * 0.8),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.22", color="k", lw=lw, shrinkA=0, shrinkB=0))


def tag(ax, x, y, tx, ty, num):
    """部件编号，带引出线。"""
    ax.plot([x, tx], [y, ty], color="k", lw=0.6)
    ax.text(tx, ty, f" {num} " if tx >= x else f"{num} ", ha="left" if tx >= x else "right", va="center", fontsize=9)


def save(fig, name):
    fig.savefig(os.path.join(HERE, name), facecolor="white")
    plt.close(fig)
    print("已生成", name)


def fig1_system():
    """系统结构：风电、储能、N台电解槽及其本地控制器，控制器连成环。"""
    fig, ax = canvas(16, 12, 100, 75)
    # 交流母线
    ax.plot([4, 96], [50, 50], color="k", lw=2.4)
    tag(ax, 60, 50, 64, 54, 3)
    # 储能
    box(ax, 6, 62, 14, 6, "储能电池")
    box(ax, 6, 53, 14, 6, "储能变流器")
    arrow(ax, [(13, 62), (13, 59)], head=False)
    arrow(ax, [(13, 53), (13, 50)], head=False)
    tag(ax, 20, 65, 23, 67, 4)
    tag(ax, 20, 56, 23, 58, 5)
    # 风电
    tx, ty = 82, 62
    ax.plot([tx, tx], [ty, ty + 7], color="k", lw=LW)
    for ang in (90, 210, 330):
        a = np.deg2rad(ang)
        ax.plot([tx, tx + 3.2 * np.cos(a)], [ty + 7, ty + 7 + 3.2 * np.sin(a)], color="k", lw=LW)
    tag(ax, tx + 1.5, ty + 9.5, tx + 5, ty + 9.5, 1)
    box(ax, 75, 53, 14, 6, "风电变流器")
    arrow(ax, [(tx, ty), (tx, 59)], head=False)
    arrow(ax, [(82, 53), (82, 50)], head=False)
    tag(ax, 89, 56, 92, 58, 2)
    # 电解槽支路
    cols = [(14, "1"), (36, "2"), (80, "N")]
    for c, k in cols:
        arrow(ax, [(c, 50), (c, 44)], head=False)
        box(ax, c - 6, 38, 12, 6, "整流电源")
        arrow(ax, [(c, 38), (c, 33)], head=False)
        box(ax, c - 6, 26, 12, 7, f"电解槽{k}")
        arrow(ax, [(c, 26), (c, 16)], ls=(0, (3, 2)))                    # 本机温度等测量
        box(ax, c - 6, 10, 12, 6, f"本地控制器{k}", fs=8)
        arrow(ax, [(c - 6, 14.5), (c - 8.5, 14.5), (c - 8.5, 41), (c - 6, 41)], ls=(0, (3, 2)))   # 负荷率设定值
    ax.text(58, 29.5, "…", ha="center", va="center", fontsize=14)
    ax.text(58, 11.5, "…", ha="center", va="center", fontsize=14)
    tag(ax, 42, 41, 46, 45, 7)
    tag(ax, 42, 29.5, 46, 33, 6)
    tag(ax, 42, 11, 46, 7.5, 8)
    # 环形通信链路，沿编号单向转发
    arrow(ax, [(20, 11.5), (30, 11.5)])
    arrow(ax, [(42, 11.5), (54, 11.5)])
    arrow(ax, [(62, 11.5), (74, 11.5)])
    arrow(ax, [(80, 10), (80, 4), (14, 4), (14, 10)])
    tag(ax, 66, 4, 66, 1.2, 9)
    # 储能信息由与储能相邻的1号控制器读取
    arrow(ax, [(6, 56), (2.5, 56), (2.5, 12.5), (8, 12.5)], ls=(0, (6, 2, 1, 2)))
    tag(ax, 2.5, 47, 5.5, 47.5, 10)
    save(fig, "图1-系统结构图.png")


def fig2_controller():
    """本地控制器内部结构。"""
    fig, ax = canvas(16, 9.6, 100, 60)
    ax.add_patch(Rectangle((14, 6), 68, 48, fill=False, ec="k", lw=LW, ls=(0, (5, 3))))
    tag(ax, 82, 50, 86, 53, 8)
    box(ax, 18, 40, 22, 9, "通信单元")
    box(ax, 18, 23, 22, 9, "本机测量单元")
    box(ax, 18, 9, 22, 7, "轮值序号单元")
    box(ax, 47, 23, 10, 15, "观测\n组合\n单元")
    box(ax, 62, 23, 10, 15, "策略\n网络")
    box(ax, 62, 9, 16, 9, "安全规则层")
    tag(ax, 40, 46, 43, 50.5, 81); tag(ax, 40, 26, 43, 22, 82); tag(ax, 40, 11, 43, 7.5, 83)
    tag(ax, 52, 23, 54.5, 19, 84); tag(ax, 62, 24.5, 59.5, 20.5, 85); tag(ax, 62, 11, 57.5, 9, 86)
    # 外部对象
    box(ax, 0.5, 40, 10, 9, "上游\n控制器", fs=8)
    box(ax, 89, 40, 10.5, 9, "下游\n控制器", fs=8)
    box(ax, 0.5, 23, 10, 9, "电解槽", fs=8)
    box(ax, 89, 9, 10.5, 9, "整流电源", fs=8)
    arrow(ax, [(10.5, 44.5), (18, 44.5)])
    arrow(ax, [(40, 47), (89, 47)])                                      # 原样转发储能信息
    arrow(ax, [(10.5, 27.5), (18, 27.5)])
    arrow(ax, [(40, 42), (44, 42), (44, 34), (47, 34)])                 # 储能电量、功率、功率变化量
    arrow(ax, [(40, 27.5), (47, 27.5)])                                 # 温度、负荷率、是否运行
    arrow(ax, [(40, 12.5), (44, 12.5), (44, 25.5), (47, 25.5)])
    arrow(ax, [(57, 30.5), (62, 30.5)])
    arrow(ax, [(67, 23), (67, 18)])
    arrow(ax, [(78, 13.5), (89, 13.5)])
    arrow(ax, [(44, 42), (44, 39.5), (75.5, 39.5), (75.5, 18)])          # 安全规则层同时读取储能信息
    ax.text(64.5, 49, "储能信息", ha="center", va="bottom", fontsize=8)
    ax.text(85.5, 15, "负荷率\n设定值", ha="center", va="bottom", fontsize=8)
    save(fig, "图2-本地控制器框图.png")


def fig3_flow():
    """训练与部署流程。"""
    fig, ax = canvas(16, 19, 100, 119)
    W, H, x0 = 52, 8, 24
    steps = [
        ("S1", "建立电解槽集群仿真环境"),
        ("S2", "各台电解槽用共享策略网络生成负荷率\n经安全规则层修正后作用于仿真环境"),
        ("S3", "计算全队共享奖励"),
        ("S4", "评价网络利用全局信息评估当前策略\n更新策略网络和评价网络"),
    ]
    ys = [106, 92, 78, 64]
    for (s, t), y in zip(steps, ys):
        box(ax, x0, y, W, H, t)
        ax.text(x0 + W + 2, y + H / 2, s, va="center")
    for a, b in zip(ys[:-1], ys[1:]):
        arrow(ax, [(50, a), (50, b + H)])
    # S5 判断
    cy = 52
    ax.add_patch(Polygon([(50, cy + 6), (68, cy), (50, cy - 6), (32, cy)], closed=True, fill=False, ec="k", lw=LW))
    ax.text(50, cy, "策略是否收敛", ha="center", va="center")
    ax.text(70, cy + 1.5, "S5", va="center")
    arrow(ax, [(50, ys[-1]), (50, cy + 6)])
    arrow(ax, [(32, cy), (14, cy), (14, ys[1] + H / 2), (x0, ys[1] + H / 2)])
    ax.text(23, cy + 1.2, "否", ha="center", va="bottom")
    ax.text(51.5, 41.8, "是", ha="left", va="center")
    deploy = [
        ("S6", "去除评价网络\n将策略网络复制到各台本地控制器"),
        ("S7", "各控制器读取本机测量量\n以及沿环转发的储能信息"),
        ("S8", "策略网络输出负荷率\n经安全规则层修正后下发至整流电源"),
    ]
    ys2 = [32, 18, 4]
    arrow(ax, [(50, cy - 6), (50, ys2[0] + H)])
    for (s, t), y in zip(deploy, ys2):
        box(ax, x0, y, W, H, t)
        ax.text(x0 + W + 2, y + H / 2, s, va="center")
    for a, b in zip(ys2[:-1], ys2[1:]):
        arrow(ax, [(50, a), (50, b + H)])
    arrow(ax, [(x0, ys2[2] + H / 2), (14, ys2[2] + H / 2), (14, ys2[1] + H / 2), (x0, ys2[1] + H / 2)])
    ax.text(12.5, (ys2[1] + ys2[2]) / 2 + H / 2, "下一\n控制周期", ha="right", va="center", fontsize=8)
    # 阶段分隔
    ax.plot([2, 98], [44.2, 44.2], color="k", lw=0.6, ls=(0, (5, 3)))
    ax.text(97, 117, "离线训练阶段", ha="right", va="top")
    ax.text(97, 42.5, "在线部署阶段", ha="right", va="top")
    save(fig, "图3-训练与部署流程图.png")


METHODS = ["均分规则", "效率优先规则", "完美预测混合整数规划", "本发明"]
HATCH = ["", "////", "....", None]   # 本发明用实心黑


def read_eval():
    vals = {}
    for line in open(os.path.join(RUN, "eval.txt"), encoding="utf-8"):
        name, _, rest = line.partition(": ")
        if "=" in rest:
            vals[name] = {k: float(v) for k, v in re.findall(r"(\w+)=([-\d.]+)", rest)}
    return vals


def fig4_compare():
    v = read_eval()
    items = [("revenue", "日收益/元", "{:.0f}"), ("starts", "日启停次数", "{:.1f}"), ("shed", "日失负荷电量/MWh", "{:.3f}")]
    fig, axs = plt.subplots(1, 3, figsize=(16 * CM, 6.5 * CM))
    for ax, (k, lab, fmt) in zip(axs, items):
        ys = [v[m][k] for m in METHODS]
        for i, y in enumerate(ys):
            h = HATCH[i]
            ax.bar(i, y, 0.62, color="k" if h is None else "white", edgecolor="k", hatch=h or "", lw=0.8,
                   label=METHODS[i])
            ax.text(i, y, fmt.format(y), ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks([])
        ax.set_title(lab, fontsize=9)
        ax.set_ylim(0, max(ys) * 1.15)
        ax.tick_params(width=0.8, labelsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, fontsize=8,
               handlelength=1.6, columnspacing=1.5)
    fig.tight_layout(w_pad=1.2, rect=(0, 0, 1, 0.9))
    save(fig, "图4-仿真结果对比图.png")


def fig5_day():
    """典型日运行曲线：选风电波动最大的测试日。"""
    d = np.load(os.path.join(RUN, "eval_data.npz"))
    wind = d["wind"]
    e = int(np.argmax(np.abs(np.diff(wind, axis=1)).sum(1)))
    t = np.arange(wind.shape[1]) * 5 / 60
    fig, axs = plt.subplots(3, 1, figsize=(16 * CM, 13 * CM), sharex=True)
    ax = axs[0]
    ax.plot(t, wind[e], color="k", lw=0.7, ls=(0, (1, 1)), label="风电功率")
    ax.plot(t, d["效率优先规则__p_el"][:, e], color="k", lw=0.9, ls=(0, (4, 2)), label="效率优先规则")
    ax.plot(t, d["本发明__p_el"][:, e], color="k", lw=1.2, label="本发明")
    ax.set_ylabel("功率/MW")
    ax.legend(loc="upper right", fontsize=7.5, frameon=False, ncol=3)
    ax = axs[1]
    ax.step(t, (d["效率优先规则__x"][:, e] > 0).sum(1), where="post", color="k", lw=0.9, ls=(0, (4, 2)))
    ax.step(t, (d["本发明__x"][:, e] > 0).sum(1), where="post", color="k", lw=1.2)
    ax.set_ylabel("运行台数")
    ax.set_yticks(range(0, 9, 2))
    ax = axs[2]
    ax.plot(t, d["效率优先规则__soc"][:, e], color="k", lw=0.9, ls=(0, (4, 2)))
    ax.plot(t, d["本发明__soc"][:, e], color="k", lw=1.2)
    for y in (0.05, 0.95):
        ax.axhline(y, color="k", lw=0.5, ls=(0, (1, 2)))
    ax.set_ylabel("储能电量")
    ax.set_ylim(0, 1)
    ax.set_xlabel("时间/h")
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 4))
    for ax, lab in zip(axs, "abc"):
        ax.text(-0.09, 1.0, f"({lab})", transform=ax.transAxes, va="top")
        ax.tick_params(width=0.8, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(h_pad=0.6)
    save(fig, "图5-典型日运行曲线.png")


if __name__ == "__main__":
    fig1_system(); fig2_controller(); fig3_flow(); fig4_compare(); fig5_day()
