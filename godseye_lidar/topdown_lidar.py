"""Top-down ("god's eye") LiDAR view.

Simulates a spinning LiDAR driving through a scene of boxes over a ground plane,
accumulates every scan in the world frame, and renders a bird's-eye height map.

To use real data, replace simulate_scan() with a reader that returns an (N, 3)
array of world-frame points plus the sensor position for each scan.
"""

import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")  # render to file; remove this line for an interactive window
import matplotlib.pyplot as plt  # noqa: E402

SENSOR_HEIGHT = 1.8  # meters above ground
MAX_RANGE = 60.0     # meters
N_AZIMUTH = 720      # horizontal samples per rotation
BEAM_ELEVATIONS = np.linspace(-15, 10, 32)  # degrees, like a 32-beam spinning LiDAR

# Axis-aligned boxes (buildings/obstacles): xmin, ymin, xmax, ymax, height
BOXES = np.array([
    [10.0, -5.0, 18.0, 3.0, 6.0],
    [-20.0, 12.0, -12.0, 20.0, 10.0],
    [25.0, 15.0, 32.0, 25.0, 4.0],
    [-15.0, -25.0, -5.0, -15.0, 8.0],
    [0.0, 30.0, 12.0, 36.0, 3.0],
    [-40.0, -10.0, -34.0, 0.0, 5.0],
])


def ray_directions():
    """Unit ray directions in the sensor frame (x forward, y left, z up)."""
    az = np.linspace(0, 2 * np.pi, N_AZIMUTH, endpoint=False)
    el = np.radians(BEAM_ELEVATIONS)
    A, E = np.meshgrid(az, el)
    d = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], axis=-1)
    return d.reshape(-1, 3)


def cast_rays(origin, dirs):
    """Return hit points for rays from origin against the ground and boxes.

    Rays that hit nothing within MAX_RANGE are dropped.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        # Ground plane z = 0
        t_ground = -origin[2] / dirs[:, 2]
        t_best = np.where((dirs[:, 2] < -1e-6) & (t_ground > 0), t_ground, np.inf)

        # Boxes: slab test in x and y, then keep the nearest entry point
        for x0, y0, x1, y1, h in BOXES:
            tx0 = (x0 - origin[0]) / dirs[:, 0]
            tx1 = (x1 - origin[0]) / dirs[:, 0]
            ty0 = (y0 - origin[1]) / dirs[:, 1]
            ty1 = (y1 - origin[1]) / dirs[:, 1]
            t_in = np.fmax(np.minimum(tx0, tx1), np.minimum(ty0, ty1))
            t_out = np.fmin(np.maximum(tx0, tx1), np.maximum(ty0, ty1))
            t_entry = t_in
            # hit if the ray enters the footprint in front of us, and the
            # entry point lies below the box top
            z_entry = origin[2] + t_entry * dirs[:, 2]
            hit = (t_out >= t_in) & (t_in > 0) & (z_entry >= 0) & (z_entry <= h)
            t_best = np.where(hit & (t_entry < t_best), t_entry, t_best)

    keep = np.isfinite(t_best) & (t_best <= MAX_RANGE)
    return origin + t_best[keep, None] * dirs[keep]


def trajectory(n_frames):
    """Sensor path: a gentle S-curve through the scene. Returns x, y, yaw."""
    x = np.linspace(-35, 35, n_frames)
    y = 4.0 * np.sin(x / 10.0)
    dy_dx = 0.4 * np.cos(x / 10.0)
    yaw = np.arctan2(dy_dx, 1.0)
    return x, y, yaw


def simulate_scan(x, y, yaw, dirs_ego, rng):
    """One rotation of the LiDAR at pose (x, y, yaw). Returns world-frame points."""
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    dirs_world = dirs_ego @ rot.T
    origin = np.array([x, y, SENSOR_HEIGHT])
    pts = cast_rays(origin, dirs_world)
    pts += rng.normal(0.0, 0.02, pts.shape)  # range noise
    return pts


def topdown_grid(points, extent, res):
    """Max-height grid. Row index = y, column index = x, origin at the grid center."""
    n = int(round(2 * extent / res))
    grid = np.full((n, n), np.nan)
    col = np.floor((points[:, 0] + extent) / res).astype(int)
    row = np.floor((points[:, 1] + extent) / res).astype(int)
    ok = (row >= 0) & (row < n) & (col >= 0) & (col < n)
    np.fmax.at(grid, (row[ok], col[ok]), points[ok, 2])
    return grid


def render(grid, trail, x_last, y_last, yaw_last, extent, out_path):
    fig, ax = plt.subplots(figsize=(9, 9), facecolor="black")
    ax.set_facecolor("black")
    im = ax.imshow(
        grid, origin="lower", extent=[-extent, extent, -extent, extent],
        cmap="terrain", vmin=0, vmax=12, interpolation="nearest",
    )
    ax.plot(trail[:, 0], trail[:, 1], color="white", lw=0.8, alpha=0.6, label="path")
    ax.annotate(
        "", xy=(x_last + 3 * np.cos(yaw_last), y_last + 3 * np.sin(yaw_last)),
        xytext=(x_last, y_last),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )
    ax.plot(x_last, y_last, marker="^", color="red", ms=10, label="ego")
    ax.set_title("God's eye LiDAR view (height in m)", color="white")
    ax.set_xlabel("x (m)", color="white")
    ax.set_ylabel("y (m)", color="white")
    ax.tick_params(colors="white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("height (m)", color="white")
    cbar.ax.tick_params(colors="white")
    ax.legend(loc="upper right", facecolor="black", labelcolor="white")
    fig.savefig(out_path, dpi=120, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=200, help="number of scans to accumulate")
    parser.add_argument("--extent", type=float, default=60.0, help="half-width of the map in meters")
    parser.add_argument("--res", type=float, default=0.25, help="grid cell size in meters")
    parser.add_argument("--out", default="godseye.png", help="output PNG path")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    dirs_ego = ray_directions()
    xs, ys, yaws = trajectory(args.frames)

    all_points = []
    for x, y, yaw in zip(xs, ys, yaws):
        all_points.append(simulate_scan(x, y, yaw, dirs_ego, rng))
    points = np.vstack(all_points)
    print(f"scans: {args.frames}  points: {len(points):,}")

    grid = topdown_grid(points, args.extent, args.res)
    filled = np.count_nonzero(~np.isnan(grid))
    print(f"grid: {grid.shape[0]}x{grid.shape[1]}  filled cells: {filled:,}")

    render(grid, np.column_stack([xs, ys]), xs[-1], ys[-1], yaws[-1], args.extent, args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
