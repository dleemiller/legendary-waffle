"""Shared preprocessing utilities for swipe path data.

This module provides a single source of truth for path preprocessing,
used by both the training dataset and the HuggingFace processor.
"""

import numpy as np


def preprocess_raw_path_to_features(
    data_points: list[dict],
    max_len: int,
    *,
    resample_mode: str = "spatial",
    dt_clamp_min_ms: float = 1.0,
    dt_clamp_max_ms: float = 200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a raw `{"x","y","t"}` path to fixed-length engineered features.

    This is the fast path used by training and the HuggingFace processor. It avoids
    building an intermediate list-of-dicts representation by:
    1) extracting x/y/t arrays once,
    2) resampling x/y using spatial- or time-uniform interpolation,
    3) recomputing dx/dy/ds and log_dt on the resampled trajectory.

    Args:
        data_points: Raw path as a list of dicts with keys: "x", "y", "t".
        max_len: Target length.
        resample_mode: "spatial" (arc-length) or "time" (cumulative dt).
        dt_clamp_min_ms: Clamp for dt feature after resampling (first dt remains 0).
        dt_clamp_max_ms: Clamp for dt feature after resampling.

    Returns:
        (features, mask) where:
          - features: [max_len, 6] float32 array (x, y, dx, dy, ds, log_dt)
          - mask: [max_len] int64 array (1 for valid; all-ones for non-empty paths)
    """
    num_points = len(data_points)
    if num_points == 0:
        return (
            np.zeros((max_len, 6), dtype=np.float32),
            np.zeros(max_len, dtype=np.int64),
        )

    x = np.fromiter((p["x"] for p in data_points), dtype=np.float64, count=num_points)
    y = np.fromiter((p["y"] for p in data_points), dtype=np.float64, count=num_points)
    t = np.fromiter((p["t"] for p in data_points), dtype=np.float64, count=num_points)

    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)

    # Per-step deltas and axes for resampling
    dx_in = np.concatenate([[0.0], np.diff(x)])
    dy_in = np.concatenate([[0.0], np.diff(y)])
    ds_in = np.hypot(dx_in, dy_in)
    dt_raw_in = np.concatenate([[0.0], np.diff(t)])

    s = np.cumsum(ds_in)
    tau = np.cumsum(dt_raw_in)

    if resample_mode not in {"spatial", "time"}:
        raise ValueError(f"Unknown resample_mode={resample_mode!r} (use 'spatial' or 'time')")

    eps = 1e-12
    if resample_mode == "time" and tau[-1] > eps:
        target_tau = np.linspace(0.0, float(tau[-1]), max_len, dtype=np.float64)
        x_r = np.interp(target_tau, tau, x)
        y_r = np.interp(target_tau, tau, y)
        tau_r = target_tau
    else:
        # Spatial sampling (or fallback when time axis is degenerate).
        if s[-1] <= eps:
            original = np.arange(num_points, dtype=np.float64)
            target = np.linspace(0, num_points - 1, max_len, dtype=np.float64)
            x_r = np.interp(target, original, x)
            y_r = np.interp(target, original, y)
            tau_r = np.interp(target, original, tau)
        else:
            target_s = np.linspace(0.0, float(s[-1]), max_len, dtype=np.float64)
            x_r = np.interp(target_s, s, x)
            y_r = np.interp(target_s, s, y)
            tau_r = np.interp(target_s, s, tau)

    dx = np.concatenate([[0.0], np.diff(x_r)])
    dy = np.concatenate([[0.0], np.diff(y_r)])
    ds = np.hypot(dx, dy)
    dt_raw_r = np.concatenate([[0.0], np.diff(tau_r)])
    dt_feat = np.clip(dt_raw_r, dt_clamp_min_ms, dt_clamp_max_ms)
    dt_feat[0] = 0.0
    log_dt = np.log1p(np.maximum(0.0, dt_feat))

    mask = np.ones(max_len, dtype=np.int64)
    features = np.stack([x_r, y_r, dx, dy, ds, log_dt], axis=-1).astype(np.float32)
    return features, mask


def normalize_and_compute_features(
    data_points: list[dict],
    dt_clamp_min_ms: float = 1.0,
    dt_clamp_max_ms: float = 200.0,
) -> list[dict]:
    """
    Normalize coordinates and compute motion features.

    Computes delta features (dx, dy, dt) and log-scaled time deltas.
    First point has dx=dy=dt=0 by convention.

    Args:
        data_points: List of {"x", "y", "t"} dicts
        dt_clamp_min_ms: Minimum dt in milliseconds (inclusive).
        dt_clamp_max_ms: Maximum dt in milliseconds (inclusive).

    Returns:
        List of dicts with keys:
          - x, y: normalized coordinates in [0, 1]
          - t: raw timestamp from input (passed through)
          - dx, dy: deltas in x/y
          - ds: sqrt(dx^2 + dy^2)
          - dt_raw: raw time delta (unclamped)
          - dt: clamped time delta used for feature stability
          - log_dt: log1p(dt)
    """
    if not data_points:
        return []

    num_points = len(data_points)
    x = np.fromiter((p["x"] for p in data_points), dtype=np.float64, count=num_points)
    y = np.fromiter((p["y"] for p in data_points), dtype=np.float64, count=num_points)
    t = np.fromiter((p["t"] for p in data_points), dtype=np.float64, count=num_points)

    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)

    dx = np.concatenate([[0.0], np.diff(x)])
    dy = np.concatenate([[0.0], np.diff(y)])
    ds = np.hypot(dx, dy)
    dt_raw = np.concatenate([[0.0], np.diff(t)])

    dt = np.clip(dt_raw, dt_clamp_min_ms, dt_clamp_max_ms)
    dt[0] = 0.0
    log_dt = np.log1p(np.maximum(0.0, dt))

    out: list[dict] = []
    for i in range(num_points):
        out.append(
            {
                "x": float(x[i]),
                "y": float(y[i]),
                "t": float(t[i]),
                "dx": float(dx[i]),
                "dy": float(dy[i]),
                "ds": float(ds[i]),
                "dt_raw": float(dt_raw[i]),
                "dt": float(dt[i]),
                "log_dt": float(log_dt[i]),
            }
        )
    return out


def sample_path_points_with_features(
    data_points: list[dict],
    max_len: int,
    *,
    resample_mode: str = "spatial",
    dt_clamp_min_ms: float = 1.0,
    dt_clamp_max_ms: float = 200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample path points with motion features to fixed length using interpolation.

    Always uses interpolation (no zero-padding) to preserve feature structure.
    Paths shorter than max_len are upsampled; longer paths are downsampled.

    Modes:
      - resample_mode="spatial": sample approximately uniformly in arc length (distance).
      - resample_mode="time": sample uniformly in time (dwell regions get more samples).

    Args:
        data_points: List of coordinate dicts. Expected keys: x, y and either:
            - dx, dy (preferred), plus optional ds, dt, log_dt; or
            - ds/log_dt/dt (ds can be derived from dx/dy; dt from log_dt).
        max_len: Target length
        resample_mode: "spatial" or "time"
        dt_clamp_min_ms: Clamp for dt feature after resampling (first dt remains 0).
        dt_clamp_max_ms: Clamp for dt feature after resampling.

    Returns:
        Tuple of (features, mask) where:
            - features: [max_len, 6] array with (x, y, dx, dy, ds, log_dt)
            - mask: [max_len] binary mask (all 1s since we always interpolate)
    """
    num_points = len(data_points)

    if num_points == 0:
        # Empty path - return zeros
        return (
            np.zeros((max_len, 6), dtype=np.float32),
            np.zeros(max_len, dtype=np.int64),
        )

    # Extract base signals
    x = np.fromiter((p["x"] for p in data_points), dtype=np.float64, count=num_points)
    y = np.fromiter((p["y"] for p in data_points), dtype=np.float64, count=num_points)

    # Prefer provided dx/dy, otherwise derive from x/y
    if all("dx" in p for p in data_points) and all("dy" in p for p in data_points):
        dx_in = np.fromiter((p["dx"] for p in data_points), dtype=np.float64, count=num_points)
        dy_in = np.fromiter((p["dy"] for p in data_points), dtype=np.float64, count=num_points)
    else:
        dx_in = np.concatenate([[0.0], np.diff(x)])
        dy_in = np.concatenate([[0.0], np.diff(y)])

    # ds can be provided or derived from dx/dy
    if all("ds" in p for p in data_points):
        ds_in = np.fromiter((p["ds"] for p in data_points), dtype=np.float64, count=num_points)
    else:
        ds_in = np.sqrt(dx_in**2 + dy_in**2)

    # Time axis for resampling: prefer dt_raw (unclamped) so "dwell" gets represented.
    if all("dt_raw" in p for p in data_points):
        dt_axis = np.fromiter(
            (p["dt_raw"] for p in data_points), dtype=np.float64, count=num_points
        )
    elif all("dt" in p for p in data_points):
        dt_axis = np.fromiter((p["dt"] for p in data_points), dtype=np.float64, count=num_points)
    elif all("log_dt" in p for p in data_points):
        log_dt_in_raw = np.fromiter(
            (p["log_dt"] for p in data_points), dtype=np.float64, count=num_points
        )
        dt_axis = np.expm1(log_dt_in_raw)
    else:
        dt_axis = np.zeros(num_points, dtype=np.float64)

    # Cumulative arc length (s) and cumulative time (tau) for resampling
    s = np.cumsum(ds_in)
    tau = np.cumsum(dt_axis)

    if resample_mode not in {"spatial", "time"}:
        raise ValueError(f"Unknown resample_mode={resample_mode!r} (use 'spatial' or 'time')")

    eps = 1e-12

    if resample_mode == "time" and tau[-1] > eps:
        target_tau = np.linspace(0.0, float(tau[-1]), max_len, dtype=np.float64)
        x_r = np.interp(target_tau, tau, x)
        y_r = np.interp(target_tau, tau, y)
        tau_r = target_tau
    else:
        # Spatial sampling (or fallback when time axis is degenerate).
        # Handle degenerate paths (zero movement): fall back to index-based interpolation
        if s[-1] <= eps:
            original = np.arange(num_points, dtype=np.float64)
            target = np.linspace(0, num_points - 1, max_len, dtype=np.float64)
            x_r = np.interp(target, original, x)
            y_r = np.interp(target, original, y)
            tau_r = np.interp(target, original, tau)
        else:
            target_s = np.linspace(0.0, float(s[-1]), max_len, dtype=np.float64)
            x_r = np.interp(target_s, s, x)
            y_r = np.interp(target_s, s, y)
            tau_r = np.interp(target_s, s, tau)

    # Recompute deltas on the resampled path for consistency
    dx = np.concatenate([[0.0], np.diff(x_r)])
    dy = np.concatenate([[0.0], np.diff(y_r)])
    ds = np.sqrt(dx**2 + dy**2)
    dt_raw_r = np.concatenate([[0.0], np.diff(tau_r)])
    dt_feat = np.clip(dt_raw_r, dt_clamp_min_ms, dt_clamp_max_ms)
    dt_feat[0] = 0.0
    log_dt = np.log1p(np.maximum(0.0, dt_feat))

    mask = np.ones(max_len, dtype=np.int64)
    features = np.stack([x_r, y_r, dx, dy, ds, log_dt], axis=-1).astype(np.float32)
    return features, mask
