"""推理期退化评测指标计算（纯标准库，CPU 可跑）。

对应 degradation_quantification_protocol.md 第二、三节。
用法：
    python tools/compute_degradation_metrics.py          # 打印示例
    from compute_degradation_metrics import nf_gap, ra, crossing_point, fd_rate

输入约定：每个 record 是一条评测结果
    {
      "level": 0.25,          # 退化档位 s，0=clean
      "condition": "night",   # 条件分组，或 "overall"
      "protocol": "P1",       # P1 污染输入 / P2 置零输入
      "fusion": 58.1,         # mAP@0.5 (OBB)
      "rgb_only": 33.2,
      "ir_only": 44.0,
      "seed": 0,
    }
真实使用时把推理结果按上述格式汇总成 list 传进 summarize()。
"""

from collections import defaultdict


def ra(corrupted, clean):
    """Resistance Ability (MultiCorrupt): 退化后性能 / 干净性能。"""
    if clean <= 0:
        raise ValueError("clean mAP must be positive")
    return corrupted / clean


def nf_gap(fusion, rgb_only, ir_only):
    """负融合间隙 NF = 融合 - 较好的单模态。负值即负融合。"""
    return fusion - max(rgb_only, ir_only)


def crossing_point(levels, nf_values):
    """NF 由正转负的临界档位 s*，在相邻两档间线性插值。"""
    if len(levels) != len(nf_values) or len(levels) < 2:
        raise ValueError("levels and nf_values must have the same length >= 2")
    for i in range(len(levels) - 1):
        a, b = nf_values[i], nf_values[i + 1]
        if a > 0 >= b:
            return levels[i] + (levels[i + 1] - levels[i]) * a / (a - b)
        if a == 0:
            return levels[i]
    return None  # 未观察到交叉 -> 走 Plan B1（比较下降斜率）


def degradation_slope(levels, values):
    """最小二乘斜率，用于 Plan B1：融合是否比单模态下降更快。"""
    n = len(levels)
    mx = sum(levels) / n
    my = sum(values) / n
    den = sum((x - mx) ** 2 for x in levels)
    if den == 0:
        raise ValueError("levels must not be constant")
    return sum((x - mx) * (y - my) for x, y in zip(levels, values)) / den


def fd_rate(detected_mono, detected_multi):
    """Fusion Degradation rate: 单模态能检出但融合漏掉的实例占比。

    detected_* 是已按 IoU 匹配到 GT 的实例 id 集合（set/tuple 均可）。
    """
    a, b = set(detected_mono), set(detected_multi)
    union = a | b
    if not union:
        return 0.0
    return len(a - b) / len(union)


def summarize(records):
    """按 (condition, protocol, level) 聚合，多 seed 报 mean±std，输出主表。"""
    buckets = defaultdict(list)
    for r in records:
        key = (r.get("condition", "overall"), r.get("protocol", "P1"), r["level"])
        buckets[key].append(nf_gap(r["fusion"], r["rgb_only"], r["ir_only"]))

    rows = []
    for (cond, proto, level), vals in sorted(buckets.items()):
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        rows.append((cond, proto, level, mean, std))
    return rows


def _mean_std(vals):
    m = sum(vals) / len(vals)
    s = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
    return m, s


def demo():
    """构造一组符合预期形状的假数据，演示主表与临界点如何产生。"""
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    fusion = [65.0, 59.0, 49.0, 33.0, 20.0]
    rgb = [45.0, 33.0, 20.0, 10.0, 5.0]
    ir = [44.0] * 5  # IR 路未被扰动，近似水平线

    records = []
    for i, s in enumerate(levels):
        records.append({
            "level": s, "condition": "overall", "protocol": "P1",
            "fusion": fusion[i], "rgb_only": rgb[i], "ir_only": ir[i], "seed": 0,
        })

    print("level   fusion  rgb_only  ir_only    NF     RA_fusion")
    for r in records:
        nf = nf_gap(r["fusion"], r["rgb_only"], r["ir_only"])
        print("%5.2f  %6.1f   %6.1f   %6.1f  %+6.1f     %.3f" % (
            r["level"], r["fusion"], r["rgb_only"], r["ir_only"], nf,
            ra(r["fusion"], fusion[0]),
        ))

    nf_values = [nf_gap(f, a, b) for f, a, b in zip(fusion, rgb, ir)]
    s_star = crossing_point(levels, nf_values)
    print("\ns* (crossing) =", "%.3f" % s_star if s_star is not None else "None -> Plan B1")
    print("slope: fusion %.1f | rgb-only %.1f | ir-only %.1f" % (
        degradation_slope(levels, fusion),
        degradation_slope(levels, rgb),
        degradation_slope(levels, ir),
    ))
    print("FD rate example =", "%.3f" % fd_rate(
        {1, 2, 3, 4, 5}, {1, 2, 5}
    ))
    print("\nmain table (condition, protocol, level, NF mean, NF std):")
    for cond, proto, level, mean, std in summarize(records):
        print("  %-8s %-3s %.2f   %+6.2f +- %.2f" % (cond, proto, level, mean, std))


if __name__ == "__main__":
    demo()
