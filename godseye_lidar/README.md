# God's Eye LiDAR View

A bird's-eye (top-down) height map built from LiDAR scans. The included script
simulates a spinning 32-beam LiDAR driving past boxes on a ground plane,
accumulates the scans in the world frame, and renders the result.

## Run

```
uv run --with numpy --with matplotlib python topdown_lidar.py
```

Options: `--frames 200`, `--extent 60` (map half-width in m), `--res 0.25` (cell size in m), `--out godseye.png`, `--seed 0`.

## Using real data

Replace `simulate_scan()` with a reader that returns world-frame `(N, 3)` points per scan.
`topdown_grid()` and `render()` take any point array, so they work unchanged.

## How it works

1. Ray-cast each beam (azimuth × elevation) against the ground plane and boxes.
2. Transform hits into the world frame using the sensor pose for each scan.
3. Bin points into a grid, keeping the max height per cell.
4. Render the grid with a colormap, the vehicle path, and the current pose.
