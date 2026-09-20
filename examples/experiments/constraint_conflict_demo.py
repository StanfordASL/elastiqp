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
    python examples/experiments/constraint_conflict_demo.py [--save fig.pdf] [--no-animate]
"""

import argparse
from dataclasses import dataclass

import elastiqp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Circle, FancyArrowPatch

# --- Environment -------------------------------------------------------------

# Room corner: the robot must stay inside {x <= WALL_X, y <= WALL_Y}
WALL_X = 1.5
WALL_Y = 1.5

ROBOT_RADIUS = 0.15
OBS_RADIUS = 0.30

ROBOT_START = np.array([1.0, 1.0])
OBS_START = np.array([-2.5, -2.2])
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
CONSTRAINT_COLORS = ["tab:orange", "tab:green", "tab:purple"]


@dataclass
class Scenario:
    name: str
    penalty: np.ndarray
    description: str


SCENARIOS = [
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
    G = np.eye(2)
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
    margin = np.linalg.norm(d) - R  # signed distance, for reporting
    return G_row, h_val, margin


def build_qp(p, v, p_obs, v_obs, u_nom):
    """Assemble (Q, q, G, h) for min 0.5||u - u_nom||^2 s.t. CBF rows G u <= h."""
    Gw, hw, wall_margins = wall_rows(p, v)
    Go, ho, obs_margin = obstacle_row(p, v, p_obs, v_obs)
    G = np.vstack([Gw, Go])
    h = np.append(hw, ho)
    margins = np.append(wall_margins, obs_margin)
    return np.eye(2), -u_nom, G, h, margins


# --- Simulation ---------------------------------------------------------------


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

    log = {k: [] for k in ("time", "p", "p_obs", "margins", "t", "z", "iters")}
    for k in range(round(SIM_TIME / DT)):
        u_nom = -KP * (p - ROBOT_START) - KD * v
        _, q, G, h, margins = build_qp(p, v, p_obs, v_obs, u_nom)
        solver.update(q=q, G=G, h=h)
        sol = solver.solve()
        if not sol.converged:
            # A rare warm-start stall at the pinch can hit MaxIter; the iterate
            # is still usable and the solver recovers on the next tick.
            print(f"  [warn] {sol.status} at t={k * DT:.2f}s (iters={sol.iters})")
        u = sol.x

        log["time"].append(k * DT)
        log["p"].append(p.copy())
        log["p_obs"].append(p_obs.copy())
        log["margins"].append(margins)
        log["t"].append(sol.t.copy())
        log["z"].append(sol.z.copy())
        log["iters"].append(sol.iters)

        # Exact double-integrator step under zero-order-hold u
        p = p + v * DT + 0.5 * u * DT**2
        v = v + u * DT
        p_obs = p_obs + v_obs * DT

    return {k: np.asarray(val) for k, val in log.items()}


# --- Plotting -----------------------------------------------------------------

TRAJ_XLIM = (0.0, 1.85)
TRAJ_YLIM = (0.0, 1.95)
TIME_WINDOW = (2.0, 6.0)
PAPER_RC = {"figure.figsize": (20, 4.25), "font.size": 14}

# Snapshot instants for the disc trails, in seconds relative to the moment the
# robot is pushed furthest from its start (which coincides with the pinch).
SNAP_OFFSETS = (-1.2, -0.6, 0.0)
SNAP_ALPHAS = (0.20, 0.42, 0.75)


def _arrow_along(ax, a, b, color, zorder=4):
    """Small arrowhead at the midpoint of segment a->b, pointing towards b."""
    mid = 0.5 * (a + b)
    step = 0.01 * (b - a) / (np.linalg.norm(b - a) + 1e-12)
    ax.annotate(
        "",
        xy=mid + step,
        xytext=mid - step,
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0, "mutation_scale": 14},
        zorder=zorder,
    )


class _HandlerArrow(HandlerPatch):
    """Legend handler drawing a dashed line with an arrowhead at the end."""

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        y = height / 2.0 - ydescent
        arrow = FancyArrowPatch(
            (-xdescent, y),
            (width - xdescent, y),
            arrowstyle="-|>",
            mutation_scale=12,
            color=orig_handle.get_edgecolor(),
            ls=orig_handle.get_linestyle(),
            lw=orig_handle.get_linewidth(),
        )
        arrow.set_transform(trans)
        return [arrow]


def _traj_legend_handles():
    """Proxy handles (dashed arrow lines) for the robot/obstacle trajectories."""
    return [
        FancyArrowPatch((0, 0), (1, 0), color=color, ls="--", lw=1.5, label=label)
        for color, label in (("tab:blue", "robot"), ("tab:red", "obstacle"))
    ]


def plot_trajectory(ax, data: dict, scenario: Scenario):
    t = data["time"]
    p = data["p"]
    p_obs = data["p_obs"]

    # Walls (room interior is down-left of the corner)
    ax.plot([WALL_X, WALL_X], [TRAJ_YLIM[0], WALL_Y], "k-", lw=2)
    ax.plot([TRAJ_XLIM[0], WALL_X], [WALL_Y, WALL_Y], "k-", lw=2)
    ax.fill_betweenx(
        [TRAJ_YLIM[0], WALL_Y], WALL_X, TRAJ_XLIM[1], color="0.85", zorder=0
    )
    ax.fill_between(TRAJ_XLIM, WALL_Y, TRAJ_YLIM[1], color="0.85", zorder=0)

    # Snapshot indices: last frame = robot's furthest excursion (the pinch),
    # earlier frames at fixed offsets so robot and obstacle discs are
    # synchronized in time.
    disp = np.linalg.norm(p - ROBOT_START, axis=1)
    i_last = int(disp.argmax())
    snap_idx = [
        int(np.clip(np.searchsorted(t, t[i_last] + off), 0, len(t) - 1))
        for off in SNAP_OFFSETS
    ]

    # Robot: path up to the pinch (the return leg retraces it), arrowhead
    # partway along the moving section.
    ax.plot(
        p[: i_last + 1, 0],
        p[: i_last + 1, 1],
        color="tab:blue",
        lw=1.5,
        ls="--",
        zorder=3,
    )
    i_mid = (snap_idx[1] + i_last) // 2
    if disp[i_last] > 0.05:
        _arrow_along(ax, p[i_mid - 1], p[i_mid + 1], "tab:blue")

    # Obstacle: dashed path starting at the first plotted disc, arrowheads in
    # the gaps between discs pointing along the motion.
    i0, i1 = snap_idx[0], snap_idx[-1]
    ax.plot(
        p_obs[i0 : i1 + 1, 0],
        p_obs[i0 : i1 + 1, 1],
        color="tab:red",
        lw=1.5,
        ls="--",
        zorder=3,
    )
    for ia, ib in zip(snap_idx[:-1], snap_idx[1:]):
        _arrow_along(ax, p_obs[ia], p_obs[ib], "tab:red")

    # Discs at the snapshot instants, fading in towards the pinch
    for i, alpha in zip(snap_idx, SNAP_ALPHAS):
        ax.add_patch(
            Circle(p_obs[i], OBS_RADIUS, color="tab:red", alpha=alpha, lw=0, zorder=1.8)
        )
        ax.add_patch(
            Circle(p[i], ROBOT_RADIUS, color="tab:blue", alpha=alpha, lw=0, zorder=2)
        )

    ax.set_xlim(*TRAJ_XLIM)
    ax.set_ylim(*TRAJ_YLIM)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(scenario.name)


def plot_paper_figure(runs):
    """Full-page-width figure: [trajectory | duals/slacks] x two scenarios."""
    with plt.rc_context(PAPER_RC):
        fig = plt.figure()
        outer = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.05, 1.0, 1.05], wspace=0.55)

        for j, (scenario, data) in enumerate(runs):
            t = data["time"]
            mask = (t >= TIME_WINDOW[0]) & (t <= TIME_WINDOW[1])

            ax_traj = fig.add_subplot(outer[0, 2 * j])
            plot_trajectory(ax_traj, data, scenario)
            # aspect="equal" leaves slack in the gridspec cell; push it away from
            # the dual/slack column so the ylabels don't collide
            ax_traj.set_anchor("W")

            zt_hspace = 0.15  # gap between z and t panels, fraction of panel height
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

            for i, (name, color) in enumerate(zip(CONSTRAINT_NAMES, CONSTRAINT_COLORS)):
                ax_z.plot(t[mask], data["z"][mask, i], color=color)
                ax_t.plot(t[mask], data["t"][mask, i], color=color, label=name)
                # Maximum attainable dual = the L1 penalty on that constraint
                ax_z.axhline(scenario.penalty[i], color=color, ls="--")

            ax_z.set_yscale("log")
            ax_z.set_ylim(1e-1, 3.0 * scenario.penalty.max())
            ax_t.set_yscale("log")
            ax_t.set_ylim(1e-1, 10.0)
            ax_z.set_xlim(*TIME_WINDOW)
            ax_z.set_ylabel("dual $z$")
            ax_z.tick_params(labelbottom=False)
            ax_t.set_ylabel("elastic slack $t$")
            ax_t.set_xlabel("time [s]")
            for ax in (ax_z, ax_t):
                ax.minorticks_off()
                ax.grid(alpha=0.3)

        # Unified legend: robot/obstacle trajectories + constraint colors
        traj_h = _traj_legend_handles()
        cons_h, cons_l = ax_t.get_legend_handles_labels()
        fig.legend(
            traj_h + cons_h,
            [h.get_label() for h in traj_h] + cons_l,
            handler_map={FancyArrowPatch: _HandlerArrow()},
            loc="lower center",
            ncols=len(traj_h) + len(cons_h),
            frameon=False,
            bbox_to_anchor=(0.5, -0.18),
        )

    return fig


def animate(runs):
    """Side-by-side animation of the scenarios, over the paper-figure trails."""
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 5))
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
    parser.add_argument("--no-animate", action="store_true", help="skip the animation")
    parser.add_argument("--save", metavar="PATH", help="save the paper figure to PATH")
    args = parser.parse_args()

    runs = []
    for scenario in SCENARIOS:
        data = simulate(scenario)
        runs.append((scenario, data))
        print(f"{scenario.name}:")
        print(f"  {scenario.description}")
        for name, w in zip(CONSTRAINT_NAMES, data["margins"].min(axis=0)):
            status = "violated" if w < 0 else "held"
            print(f"  {name:9s}: worst margin {w:+.3f} m ({status})")
        print(f"  mean solver iters: {data['iters'].mean():.1f}")

    fig = plot_paper_figure(runs)
    if args.save:
        fig.savefig(args.save, bbox_inches="tight", dpi=300)
        print(f"Saved paper figure to {args.save}")

    anim = animate(runs) if not args.no_animate else None  # noqa: F841 (keep alive)
    plt.show()


if __name__ == "__main__":
    main()
