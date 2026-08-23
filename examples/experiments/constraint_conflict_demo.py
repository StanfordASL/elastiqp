"""Constraint conflict experiment: infeasible safety constraints, reasonable control.

A 2D double-integrator "point robot" (radius r) sits near the corner of a room
formed by two wall half-plane constraints. A dynamic obstacle (also a disc)
moves at constant velocity towards the corner, squeezing the robot until
satisfying all safety constraints (2 walls + obstacle avoidance) is impossible.

Each safety constraint is written as a relative-degree-2 CBF condition on the
control input and handed to ElastiQP as an elastic inequality. When the
constraints conflict, the L1 elastic penalties decide which constraint yields:
high wall / low obstacle penalty pins the robot in the corner and accepts the
obstacle contact, while low wall / high obstacle penalty lets the robot back
through the wall slightly to keep clear of the obstacle. A hard-constrained QP
would simply return "infeasible" at the pinch and leave the controller with no
input at all.

Run (from the repo root, inside the venv):
    python examples/experiments/constraint_conflict_demo.py [--animate]
"""

import argparse
from dataclasses import dataclass, field

import elastiqp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# --- Environment -------------------------------------------------------------

# Room corner: the robot must stay inside {x <= WALL_X, y <= WALL_Y}
WALL_X = 1.5
WALL_Y = 1.5

ROBOT_RADIUS = 0.15
OBS_RADIUS = 0.30

ROBOT_START = np.array([1.0, 1.0])
OBS_START = np.array([-2.5, -2.2])  # slight asymmetry so the pinch isn't symmetric
OBS_SPEED = 1.1
# Constant velocity aimed at the corner (small offset keeps it off the diagonal)
OBS_TARGET = np.array([WALL_X + 0.1, WALL_Y - 0.1])

DT = 0.01
SIM_TIME = 7.0

# PD gains for the nominal go-to-start controller
KP = 4.0
KD = 3.0

# Class-K gains for the cascaded relative-degree-2 CBF condition:
#   h_ddot + (ALPHA1 + ALPHA2) h_dot + ALPHA1 * ALPHA2 * h >= 0
ALPHA1 = 3.0
ALPHA2 = 3.0

CONSTRAINT_NAMES = ["wall x", "wall y", "obstacle"]


@dataclass
class Scenario:
    name: str
    penalty: np.ndarray  # per-constraint L1 weight: [wall x, wall y, obstacle]
    description: str


SCENARIOS = [
    Scenario(
        "Equal penalties",
        np.array([1e3, 1e3, 1e3]),
        "No stated priority: the solver yields whichever constraint is cheapest to violate",
    ),
    Scenario(
        "Walls hard, obstacle soft",
        np.array([1e2, 1e2, 1e1]),
        "Robot stops in the corner and accepts contact with the obstacle",
    ),
    Scenario(
        "Obstacle hard, walls soft",
        np.array([1e1, 1e1, 1e2]),
        "Robot yields the workspace bounds slightly to stay clear of the obstacle",
    ),
]


# --- CBF constraint rows ------------------------------------------------------
#
# Double integrator: p_dot = v, v_dot = u. Every barrier h(p) below has relative
# degree 2, so we enforce the cascaded condition
#     h_ddot + (a1 + a2) h_dot + a1 a2 h >= 0
# which is affine in u and becomes one row of G u <= h_vec.


def wall_rows(p, v):
    """CBF rows for the two walls. Returns (G, h_vec, margins).

    Wall x barrier: h = WALL_X - px - r  (>= 0 inside the room)
        h_dot = -vx, h_ddot = -ux  =>  ux <= a1 a2 h - (a1 + a2) vx
    and symmetrically for y.
    """
    hx = WALL_X - p[0] - ROBOT_RADIUS
    hy = WALL_Y - p[1] - ROBOT_RADIUS
    a_sum, a_prod = ALPHA1 + ALPHA2, ALPHA1 * ALPHA2
    G = np.array([[1.0, 0.0], [0.0, 1.0]])
    h_vec = np.array([a_prod * hx - a_sum * v[0], a_prod * hy - a_sum * v[1]])
    return G, h_vec, np.array([hx, hy])


def obstacle_row(p, v, p_obs, v_obs):
    """CBF row for the moving obstacle. Returns (G_row, h_val, margin).

    Squared-distance barrier (smooth everywhere):
        h = ||d||^2 - R^2,  d = p - p_obs,  R = r + r_obs
        h_dot  = 2 d . (v - v_obs)
        h_ddot = 2 ||v - v_obs||^2 + 2 d . u   (obstacle acceleration = 0)
    Condition: -2 d . u <= 2 ||v - v_obs||^2 + (a1 + a2) h_dot + a1 a2 h
    """
    d = p - p_obs
    dv = v - v_obs
    R = ROBOT_RADIUS + OBS_RADIUS
    h = d @ d - R * R
    h_dot = 2.0 * d @ dv
    a_sum, a_prod = ALPHA1 + ALPHA2, ALPHA1 * ALPHA2
    G_row = -2.0 * d
    h_val = 2.0 * dv @ dv + a_sum * h_dot + a_prod * h
    margin = np.linalg.norm(d) - R  # signed distance, for plotting
    return G_row, h_val, margin


def build_qp(p, v, p_obs, v_obs, u_nom):
    """Assemble (Q, q, G, h) for min 0.5||u - u_nom||^2 s.t. CBF rows G u <= h."""
    Gw, hw, wall_margins = wall_rows(p, v)
    Go, ho, obs_margin = obstacle_row(p, v, p_obs, v_obs)
    G = np.vstack([Gw, Go])
    h = np.append(hw, ho)
    Q = np.eye(2)
    q = -u_nom
    margins = np.append(wall_margins, obs_margin)
    return Q, q, G, h, margins


# --- Simulation ---------------------------------------------------------------


@dataclass
class Log:
    time: list = field(default_factory=list)
    p: list = field(default_factory=list)
    p_obs: list = field(default_factory=list)
    u: list = field(default_factory=list)
    u_nom: list = field(default_factory=list)
    margins: list = field(
        default_factory=list
    )  # signed distances [wall x, wall y, obs]
    cbf_slack: list = field(default_factory=list)  # G u - h per row (constraint room)
    t_elastic: list = field(default_factory=list)  # ElastiQP slacks t
    z: list = field(default_factory=list)  # ElastiQP inequality duals
    iters: list = field(default_factory=list)

    def as_arrays(self):
        return {k: np.asarray(getattr(self, k)) for k in self.__dataclass_fields__}


def simulate(scenario: Scenario) -> dict:
    p = ROBOT_START.astype(float).copy()
    v = np.zeros(2)
    p_obs = OBS_START.astype(float).copy()
    v_obs = (
        OBS_SPEED * (OBS_TARGET - OBS_START) / np.linalg.norm(OBS_TARGET - OBS_START)
    )

    solver = elastiqp.Solver()
    Q, q, G, h, _ = build_qp(p, v, p_obs, v_obs, np.zeros(2))
    solver.setup(Q, q, G, h, scenario.penalty)

    log = Log()
    n_steps = int(round(SIM_TIME / DT))
    for k in range(n_steps):
        u_nom = -KP * (p - ROBOT_START) - KD * v
        _, q, G, h, margins = build_qp(p, v, p_obs, v_obs, u_nom)
        solver.update(q=q, G=G, h=h)
        sol = solver.solve()
        if not sol.converged:
            # A rare warm-start stall at the pinch can hit MaxIter; the iterate
            # is still usable and the solver recovers on the next tick.
            print(f"  [warn] {sol.status} at t={k * DT:.2f}s (iters={sol.iters})")
        u = sol.x

        log.time.append(k * DT)
        log.p.append(p.copy())
        log.p_obs.append(p_obs.copy())
        log.u.append(u.copy())
        log.u_nom.append(u_nom.copy())
        log.margins.append(margins)
        log.cbf_slack.append(G @ u - h)
        log.t_elastic.append(sol.t.copy())
        log.z.append(sol.z_ineq.copy())
        log.iters.append(sol.iters)

        # Exact double-integrator step under zero-order-hold u
        p = p + v * DT + 0.5 * u * DT**2
        v = v + u * DT
        p_obs = p_obs + v_obs * DT

    return log.as_arrays()


# --- Plotting -----------------------------------------------------------------


def plot_trajectory(
    ax, data: dict, scenario: Scenario, xlim=None, ylim=None, legend=True
):
    t = data["time"]
    p = data["p"]
    p_obs = data["p_obs"]

    # Walls (room interior is down-left of the corner)
    xlim = xlim if xlim is not None else (-3.0, 2.5)
    ylim = ylim if ylim is not None else (-3.0, 2.5)
    lim_lo = min(xlim[0], ylim[0])
    lim_hi = max(xlim[1], ylim[1])
    ax.plot([WALL_X, WALL_X], [lim_lo, WALL_Y], "k-", lw=2)
    ax.plot([lim_lo, WALL_X], [WALL_Y, WALL_Y], "k-", lw=2)
    ax.fill_betweenx([lim_lo, WALL_Y], WALL_X, lim_hi, color="0.85", zorder=0)
    ax.fill_between([lim_lo, lim_hi], WALL_Y, lim_hi, color="0.85", zorder=0)

    ax.plot(p[:, 0], p[:, 1], color="tab:blue", lw=1.5, label="robot")
    ax.plot(
        p_obs[:, 0], p_obs[:, 1], color="tab:red", lw=1.5, ls="--", label="obstacle"
    )

    # Discs at a few snapshots
    for frac in (0.0, 0.35, 0.5, 0.65, 1.0):
        i = min(int(frac * (len(t) - 1)), len(t) - 1)
        alpha = 0.15 + 0.5 * frac
        ax.add_patch(Circle(p[i], ROBOT_RADIUS, color="tab:blue", alpha=alpha, lw=0))
        ax.add_patch(Circle(p_obs[i], OBS_RADIUS, color="tab:red", alpha=alpha, lw=0))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(scenario.name)
    if legend:
        ax.legend(
            loc="lower left",
            handlelength=1.4,
            borderpad=0.3,
            labelspacing=0.3,
        )


def plot_timeseries(fig, axes, data: dict, scenario: Scenario):
    t = data["time"]
    colors = ["tab:orange", "tab:green", "tab:red"]

    ax = axes[0]
    for i, name in enumerate(CONSTRAINT_NAMES):
        ax.plot(t, data["margins"][:, i], color=colors[i], label=name)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylabel("signed distance [m]")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    for i, name in enumerate(CONSTRAINT_NAMES):
        ax.plot(t, data["t_elastic"][:, i], color=colors[i], label=name)
    ax.set_ylabel("elastic slack t")

    ax = axes[2]
    for i, name in enumerate(CONSTRAINT_NAMES):
        ax.plot(t, data["z"][:, i], color=colors[i], label=name)
    ax.set_ylabel("dual z")
    ax.set_yscale("symlog", linthresh=1.0)

    ax = axes[3]
    ax.plot(t, data["u"][:, 0], color="tab:blue", label="$u_x$")
    ax.plot(t, data["u"][:, 1], color="tab:cyan", label="$u_y$")
    ax.plot(t, data["u_nom"][:, 0], color="tab:blue", ls=":", lw=1, label="$u_x$ nom")
    ax.plot(t, data["u_nom"][:, 1], color="tab:cyan", ls=":", lw=1, label="$u_y$ nom")
    ax.set_ylabel("control [m/s²]")
    ax.set_xlabel("time [s]")
    ax.legend(fontsize=8, loc="upper right", ncols=2)

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"{scenario.name} — {scenario.description}", fontsize=10)


PAPER_SCENARIOS = ["Walls hard, obstacle soft", "Obstacle hard, walls soft"]
TRAJ_XLIM = (0.0, 1.6)
TRAJ_YLIM = (0.0, 1.8)
TIME_WINDOW = (2.0, 6.0)

# Sized for ieeeconf: figure* spans \textwidth (~7.16 in), body text is 10 pt,
# so 8 pt labels / 7 pt ticks match typical IEEE figure typography at 1:1 scale.
# Include with \includegraphics[width=\textwidth] (no scaling) to keep fonts true.
PAPER_RC = {
    "figure.figsize": (20, 4.25),
    "font.size": 14,
    # "axes.labelsize": 10,
    # "axes.titlesize": 10,
    # "xtick.labelsize": 10,
    # "ytick.labelsize": 10,
    # "legend.fontsize": 6.5,
    # "lines.linewidth": 1,
    # "axes.linewidth": 0.6,
    # "grid.linewidth": 0.4,
}


def plot_paper_figure(runs):
    """Full-page-width figure: [trajectory | duals/slacks] x two scenarios."""
    selected = [(s, d) for s, d in runs if s.name in PAPER_SCENARIOS]
    colors = ["tab:orange", "tab:green", "tab:red"]

    with plt.rc_context(PAPER_RC):
        fig = plt.figure()
        outer = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.05, 1.0, 1.05], wspace=0.55)

        for j, (scenario, data) in enumerate(selected):
            t = data["time"]
            mask = (t >= TIME_WINDOW[0]) & (t <= TIME_WINDOW[1])

            ax_traj = fig.add_subplot(outer[0, 2 * j])
            plot_trajectory(
                ax_traj, data, scenario, xlim=TRAJ_XLIM, ylim=TRAJ_YLIM, legend=False
            )
            # aspect="equal" leaves slack in the gridspec cell; push it away from
            # the dual/slack column so the ylabels don't collide
            ax_traj.set_anchor("W")

            zt_hspace = (
                0.15  # gap between z and t panels, as a fraction of panel height
            )
            inner = outer[0, 2 * j + 1].subgridspec(2, 1, hspace=zt_hspace)
            ax_z = fig.add_subplot(inner[0])
            ax_t = fig.add_subplot(inner[1], sharex=ax_z)

            # aspect="equal" shrinks the trajectory box at draw time; resolve it
            # now and pin the z/t stack to span exactly the same vertical extent
            ax_traj.apply_aspect()
            tp = ax_traj.get_position()
            zp = ax_z.get_position()
            panel_h = tp.height / (2.0 + zt_hspace)
            ax_t.set_position([zp.x0, tp.y0, zp.width, panel_h])
            ax_z.set_position(
                [zp.x0, tp.y0 + panel_h * (1.0 + zt_hspace), zp.width, panel_h]
            )

            for i, name in enumerate(CONSTRAINT_NAMES):
                ax_z.plot(t[mask], data["z"][mask, i], color=colors[i])
                ax_t.plot(
                    t[mask], data["t_elastic"][mask, i], color=colors[i], label=name
                )
                # Maximum attainable dual = the L1 penalty on that constraint
                ax_z.axhline(scenario.penalty[i], color=colors[i], ls="--")  # , lw=0.8)

            rho_max = scenario.penalty.max()
            ax_z.set_yscale("log")
            ax_z.minorticks_off()
            ax_z.set_ylim(1e-1, 3.0 * rho_max)
            ax_t.set_yscale("log")
            ax_t.set_ylim(1e-1, 10.0)
            ax_t.minorticks_off()
            # ax_z.annotate(
            #     r"$z_{\max} = \rho$",
            #     xy=(TIME_WINDOW[0], rho_max),
            #     xytext=(2, -2),
            #     textcoords="offset points",
            #     va="top",
            #     fontsize=7,
            #     color="0.3",
            # )

            ax_z.set_xlim(*TIME_WINDOW)
            ax_z.set_ylabel("dual $z$")
            ax_z.tick_params(labelbottom=False)
            ax_t.set_ylabel("elastic slack $t$")
            ax_t.set_xlabel("time [s]")
            if j == 0:
                # Unified legend: robot/obstacle trajectories + constraint colors
                traj_h, traj_l = ax_traj.get_legend_handles_labels()
                cons_h, cons_l = ax_t.get_legend_handles_labels()
                legend_handles = traj_h + cons_h
                legend_labels = traj_l + cons_l
            for ax in (ax_z, ax_t):
                ax.grid(alpha=0.3)

        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncols=len(legend_handles),
            frameon=False,
            bbox_to_anchor=(0.5, -0.18),
        )

    return fig


def animate(runs):
    """Side-by-side animation of all scenarios."""
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 5))
    if len(runs) == 1:
        axes = [axes]
    artists = []
    for ax, (scenario, data) in zip(axes, runs):
        plot_trajectory(ax, data, scenario)
        robot = Circle(data["p"][0], ROBOT_RADIUS, color="tab:blue", zorder=5)
        obs = Circle(data["p_obs"][0], OBS_RADIUS, color="tab:red", zorder=5)
        ax.add_patch(robot)
        ax.add_patch(obs)
        artists.append((robot, obs))

    n = len(runs[0][1]["time"])
    stride = 4  # 25 fps at dt=0.01

    def update(frame):
        i = min(frame * stride, n - 1)
        out = []
        for (robot, obs), (_, data) in zip(artists, runs):
            robot.center = data["p"][i]
            obs.center = data["p_obs"][i]
            out += [robot, obs]
        return out

    anim = FuncAnimation(fig, update, frames=n // stride + 1, interval=40, blit=True)
    fig.tight_layout()
    return anim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--animate", action="store_true", help="show an animation")
    parser.add_argument(
        "--detail",
        action="store_true",
        help="also show the per-scenario diagnostic plots",
    )
    parser.add_argument("--save", metavar="PATH", help="save the paper figure to PATH")
    args = parser.parse_args()

    runs = []
    for scenario in SCENARIOS:
        data = simulate(scenario)
        runs.append((scenario, data))
        worst = data["margins"].min(axis=0)
        print(f"{scenario.name}:")
        print(f"  {scenario.description}")
        for name, w in zip(CONSTRAINT_NAMES, worst):
            status = "violated" if w < 0 else "held"
            print(f"  {name:9s}: worst margin {w:+.3f} m ({status})")
        print(f"  mean solver iters: {data['iters'].mean():.1f}")

    paper_fig = plot_paper_figure(runs)
    if args.save:
        paper_fig.savefig(args.save, bbox_inches="tight", dpi=300)
        print(f"Saved paper figure to {args.save}")

    if args.detail:
        # Trajectories side by side
        fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 5))
        for ax, (scenario, data) in zip(np.atleast_1d(axes), runs):
            plot_trajectory(ax, data, scenario)
        fig.tight_layout()

        # Time series per scenario
        for scenario, data in runs:
            fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
            plot_timeseries(fig, axes, data, scenario)
            fig.tight_layout()

    anim = animate(runs) if args.animate else None  # noqa: F841 (keep alive)
    plt.show()


if __name__ == "__main__":
    main()
