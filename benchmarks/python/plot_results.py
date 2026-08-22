#!/usr/bin/env python
"""Plots for the benchmark CSVs in results/ (run the C++ benchmarks with
--csv results/ and run_diff_experiment.py first; missing CSVs are skipped).

Conventions: one fixed color per solver across every figure (color follows
the entity), line style / tint distinguishes variants (cold/warm,
forward/gradient), log axes get dots-with-error-bars rather than bars (bar
length is meaningless on a log scale), grids stay recessive.
"""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import bench_common as bc  # noqa: E402

# Categorical palette (validated, fixed assignment).
COLORS = {
    "elastiqp": "#2a78d6",
    "piqp": "#eb6834",
    "proxqp": "#1baf7a",
    "qpax": "#eda100",
}
# Lighter tint of the entity color, for the cold/warm pairing in bar and
# line charts (dot plots use open vs filled markers instead).
TINTS = {
    "elastiqp": "#9cc1ea",
}
GRAY = "#6b6a5f"
SOLVER_OF = {  # route name -> palette entity
    "elastiqp": "elastiqp", "elastiqp-warm": "elastiqp",
    "piqp-hard": "piqp", "piqp-slack": "piqp", "piqp-expanded": "piqp",
    "proxqp-hard": "proxqp", "proxqp-clfeas": "proxqp",
    "proxqp-slack": "proxqp",
    "piqp": "piqp", "proxqp": "proxqp",
    "qpax-hard": "qpax", "qpax-elastic": "qpax",
}


def read_csv(name):
    path = bc.RESULTS_DIR / name
    if not path.exists():
        print(f"skip: {name} not found")
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def style(ax):
    ax.grid(True, axis="x", color="#e5e4dc", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=9)


def save(fig, name):
    out = bc.RESULTS_DIR / name
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------- robot cold/warm timing

def plot_robot_control():
    rows = read_csv("robot_control_summary.csv")
    if not rows:
        return
    scenarios = sorted({r["scenario"] for r in rows},
                       key=lambda s: float(next(
                           r["mean_us"] for r in rows if r["scenario"] == s)))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(9.5, 2.0))
    for ax, sc in zip(np.atleast_1d(axes), scenarios):
        labels, means, stds, colors = [], [], [], []
        for mode in ("warm", "cold"):
            r = next((r for r in rows
                      if r["scenario"] == sc and r["mode"] == mode), None)
            if r is None:
                continue
            labels.append(mode)
            means.append(float(r["mean_us"]))
            stds.append(float(r["std_us"]))
            colors.append(TINTS["elastiqp"] if mode == "warm"
                          else COLORS["elastiqp"])
        ax.barh(labels, means, xerr=stds, height=0.6, color=colors,
                error_kw=dict(ecolor=GRAY, lw=1), zorder=2)
        xmax = max(means) * 1.15
        for lab, v in zip(labels, means):
            # Long bars get their label inside (the panel edge is close).
            if v > 0.55 * xmax:
                ax.text(v - xmax * 0.03, lab, f"{v:.1f} µs", va="center",
                        ha="right", fontsize=8.5, color="white", zorder=3)
            else:
                ax.text(v + xmax * 0.03, lab, f"{v:.1f} µs", va="center",
                        fontsize=8.5, color="#40403a")
        r0 = next(r for r in rows if r["scenario"] == sc)
        ax.set_title(f"{sc}  (n={r0['n']}, m={r0['m']}, p={r0['p']})",
                     fontsize=10)
        ax.set_xlim(0, xmax)
        style(ax)
    fig.subplots_adjust(wspace=0.4)
    fig.suptitle("ElastiQP per-tick solve time on robot control loops "
                 "(mean ± std, eps=1e-6)", fontsize=11, y=1.12)
    save(fig, "robot_control_timing.png")


# ------------------------------------------------ warm-start tick trace

def plot_tick_trace():
    rows = read_csv("robot_control_ticks.csv")
    if not rows:
        return
    sc = "hum-wbc"
    sub = [r for r in rows if r["scenario"] == sc]
    if not sub:
        return
    fig, ax = plt.subplots(figsize=(7, 2.8))
    for mode, col, z in (("cold", TINTS["elastiqp"], 2),
                         ("warm", COLORS["elastiqp"], 3)):
        sel = [r for r in sub if r["mode"] == mode]
        ticks = [int(r["tick"]) for r in sel]
        us = [float(r["us"]) for r in sel]
        ax.plot(ticks, us, color=col, lw=1.2, label=mode, zorder=z)
    ax.set_yscale("log")
    ax.set_xlabel("control tick", fontsize=9)
    ax.set_ylabel("solve time [µs]", fontsize=9)
    ax.set_title(f"{sc}: per-tick solve time, cold vs warm-started "
                 "(eps=1e-6)", fontsize=10)
    ax.grid(True, color="#e5e4dc", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    save(fig, "robot_control_ticks.png")


# --------------------------------------------- multisolver comparison

def plot_multisolver():
    rows = read_csv("robot_multisolver.csv")
    if not rows:
        return
    order = ["elastiqp", "elastiqp-warm", "piqp-hard", "piqp-slack",
             "proxqp-hard", "proxqp-clfeas", "proxqp-slack"]
    scenarios = ["diff-ik", "arm-osc", "biman-ik", "hum-wbc"]
    scenarios = [s for s in scenarios if any(r["scenario"] == s for r in rows)]
    fig, axes = plt.subplots(1, len(scenarios),
                             figsize=(3.2 * len(scenarios), 3.2),
                             sharey=True)
    for ax, sc in zip(np.atleast_1d(axes), scenarios):
        for i, route in enumerate(order):
            y = -i
            for variant, filled in (("feasible", False), ("conflict", True)):
                r = next((r for r in rows if r["scenario"] == sc and
                          r["route"] == route and r["variant"] == variant),
                         None)
                if r is None:
                    continue
                mean = float(r["mean_us"])
                col = COLORS[SOLVER_OF[route]]
                usable = int(r["solved"]) > 0
                marker = "o" if usable else "x"
                ax.plot(mean, y + (0.18 if variant == "feasible" else -0.18),
                        marker, color=col, markersize=7,
                        markerfacecolor=(col if filled else "white"),
                        markeredgewidth=1.6, zorder=3)
        ax.set_xscale("log")
        ax.set_title(sc, fontsize=10)
        style(ax)
    axes = np.atleast_1d(axes)
    axes[0].set_yticks([-i for i in range(len(order))])
    axes[0].set_yticklabels(order, fontsize=9)
    axes[len(scenarios) // 2].set_xlabel(
        "per-tick time [µs] — open = feasible variant, filled = conflict "
        "variant, × = no usable solution", fontsize=9)
    fig.suptitle("Robot control replay, feasible + conflict variants: cost "
                 "per solver route (eps=1e-6)", fontsize=11, y=1.04)
    save(fig, "robot_multisolver.png")


# --------------------------------------------- Maros-Meszaros profile

def _mm_profile(rows, routes, route_key, time_key, title, outname, styles):
    """Dolan-More performance profile over one Maros-Meszaros CSV."""
    routes = [s for s in routes if any(r[route_key] == s for r in rows)]
    problems = sorted({r["name"] for r in rows})
    times = {}
    for r in rows:
        t = float(r[time_key]) if r[time_key] not in ("nan", "") else np.inf
        ok = r["ok"] == "1"
        times[(r["name"], r[route_key])] = t if ok else np.inf
    best = {p: min(times.get((p, s), np.inf) for s in routes)
            for p in problems}
    taus = np.logspace(0, 4, 400)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for s in routes:
        ratios = np.array([times.get((p, s), np.inf) / best[p]
                           for p in problems if np.isfinite(best[p])])
        frac = [(ratios <= tau).mean() for tau in taus]
        col = COLORS[SOLVER_OF.get(s, s)]
        # Legend only: several curves saturate at 1.0, so end-of-line
        # labels would collide there.
        ax.step(taus, frac, styles.get(s, "-"), where="post", color=col,
                lw=2, label=s)
    ax.set_xscale("log")
    ax.set_xlim(1, 3e4)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("performance ratio τ (time / best route's time)",
                  fontsize=9)
    ax.set_ylabel("fraction of problems solved", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(True, color="#e5e4dc", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    save(fig, outname)


def plot_mm_profile():
    # C++ harness (n<=200 subset; bench_maros_meszaros --csv results/).
    rows = read_csv("maros_meszaros_results.csv")
    if rows:
        _mm_profile(rows,
                    ["elastiqp", "piqp-hard", "proxqp-hard", "piqp-expanded"],
                    "route", "time_us",
                    "Maros-Meszaros small dense subset: performance profile "
                    "(eps=1e-6)",
                    "maros_meszaros_profile.png",
                    {"piqp-expanded": "--"})
    # Python harness (paper Table III subset; run_maros_meszaros.py) --
    # includes the qpax routes.
    rows = read_csv("maros_meszaros_py_results.csv")
    if rows:
        _mm_profile(rows,
                    ["elastiqp", "piqp", "proxqp", "qpax-elastic",
                     "qpax-hard"],
                    "solver", "time_ms",
                    "Maros-Meszaros robotics-relevant subset: performance "
                    "profile (eps=1e-6)",
                    "maros_meszaros_py_profile.png",
                    {"qpax-hard": "--"})


# --------------------------------------------------- diff timing lines

def plot_diff_timing():
    rows = read_csv("diff_timing.csv")
    if not rows:
        return
    rows = sorted(rows, key=lambda r: int(r["n"]) + int(r["p"]))
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = [int(r["n"]) + int(r["p"]) for r in rows]
    series = [
        ("elqp_fwd_us", "elastiqp forward", COLORS["elastiqp"], "-"),
        ("elqp_valgrad_us", "elastiqp value+grad", COLORS["elastiqp"], "--"),
        ("naive_valgrad_us", "dense-KKT value+grad", GRAY, "--"),
        ("qpax_fwd_us", "qpax forward", COLORS["qpax"], "-"),
        ("qpax_valgrad_us", "qpax value+grad", COLORS["qpax"], "--"),
    ]
    for key, label, col, ls in series:
        yv = [float(r[key]) for r in rows]
        ax.plot(x, yv, ls, color=col, lw=2, marker="o", markersize=5,
                label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("problem size n + p (elastic form)", fontsize=9)
    ax.set_ylabel("wall time [µs]", fontsize=9)
    ax.set_title("Differentiation cost: elastiqp (condensed backward) vs "
                 "qpax (expanded formulation), jitted CPU", fontsize=10)
    ax.grid(True, color="#e5e4dc", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5)
    save(fig, "diff_timing.png")


# --------------------------------------- relax cost at robot scale

def plot_diff_robot():
    rows = read_csv("diff_robot_ticks.csv")
    if not rows:
        return
    sc, kappa = "hum-wbc", "0.001"
    sub = [r for r in rows if r["scenario"] == sc and r["kappa"] == kappa]
    if not sub:
        return
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(9.5, 2.8), gridspec_kw={"width_ratios": [2.2, 1]})
    ticks = [int(r["tick"]) for r in sub]
    us = [float(r["relax_us"]) for r in sub]
    ax.plot(ticks, us, color=COLORS["elastiqp"], lw=1.2, label="relax(κ)",
            zorder=2)
    ax.set_yscale("log")
    ax.set_xlabel("control tick", fontsize=9)
    ax.set_ylabel("relax(κ) time [µs]", fontsize=9)
    ax.set_title(f"{sc}, κ={kappa}: per-tick relax cost", fontsize=10)
    ax.grid(True, color="#e5e4dc", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)

    it = [int(r["relax_iters"]) for r in sub]
    bins = np.arange(min(it) - 0.5, max(it) + 1.5)
    ax2.hist(it, bins=bins, color=TINTS["elastiqp"], zorder=2)
    ax2.set_xlabel("relax Newton steps / tick", fontsize=9)
    ax2.set_title("iteration counts", fontsize=10)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    save(fig, "diff_robot_relax.png")


def main():
    bc.RESULTS_DIR.mkdir(exist_ok=True)
    plot_robot_control()
    plot_tick_trace()
    plot_multisolver()
    plot_mm_profile()
    plot_diff_timing()
    plot_diff_robot()


if __name__ == "__main__":
    main()
