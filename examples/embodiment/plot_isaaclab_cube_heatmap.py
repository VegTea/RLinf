#!/usr/bin/env python

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import torch


DEFAULT_CMAP = "viridis"
FIG_BG = "#f6f3ee"
AX_BG = "#fffdfa"
GRID_COLOR = "#d8d1c7"
TEXT_COLOR = "#2f261d"
SUCCESS_POINT_COLOR = "#fb8500"
FAIL_POINT_COLOR = "#264653"
DENSITY_LINE_COLOR = "#6d6875"


def _parse_range(value):
    if value is None:
        return None
    parts = [float(part) for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected min,max range, got: {value}")
    return parts


def _get_cube_positions(raw_metrics, cube_name):
    if cube_name in {"cube_1", "cube_2", "cube_3"}:
        key = f"init_{cube_name}_pos"
        if key in raw_metrics:
            return raw_metrics[key]

    if "init_cube_positions" not in raw_metrics:
        raise KeyError(
            "Cannot find init cube positions. Expected init_cube_positions or "
            "init_cube_{1,2,3}_pos in eval_metrics.pt."
        )

    cube_idx = {"cube_1": 0, "cube_2": 1, "cube_3": 2}[cube_name]
    return raw_metrics["init_cube_positions"][:, cube_idx]


def _infer_range(values, margin_ratio=0.05):
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if low == high:
        return [low - 0.01, high + 0.01]
    margin = (high - low) * margin_ratio
    return [low - margin, high + margin]


def _gaussian_kernel1d(sigma):
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def _gaussian_filter2d(values, sigma):
    if sigma <= 0:
        return values
    kernel = _gaussian_kernel1d(sigma)
    smoothed = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"), axis=0, arr=values
    )
    smoothed = np.apply_along_axis(
        lambda col: np.convolve(col, kernel, mode="same"), axis=1, arr=smoothed
    )
    return smoothed


def _setup_plot_style():
    plt.rcParams.update(
        {
            "figure.facecolor": FIG_BG,
            "axes.facecolor": AX_BG,
            "savefig.facecolor": FIG_BG,
            "axes.edgecolor": GRID_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "figure.titlesize": 16,
        }
    )


def _style_axis(ax):
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(1.2)
    ax.grid(True, color=GRID_COLOR, alpha=0.45, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _draw_rate_panel(
    ax,
    rate,
    smooth_total,
    xbins,
    ybins,
    x,
    y,
    success,
    cmap,
    show_points,
):
    rate_t = rate.T
    density_t = smooth_total.T
    valid = np.isfinite(rate_t)
    if np.any(valid):
        finite_density = density_t[valid]
        alpha = np.zeros_like(rate_t, dtype=np.float64)
        density_max = float(finite_density.max())
        if density_max > 0:
            alpha[valid] = np.clip((finite_density / density_max) ** 0.55, 0.18, 0.95)
        else:
            alpha[valid] = 0.9
    else:
        alpha = np.zeros_like(rate_t, dtype=np.float64)

    mesh = ax.pcolormesh(
        xbins,
        ybins,
        np.ma.masked_invalid(rate_t),
        shading="auto",
        cmap=cmap,
        norm=colors.Normalize(vmin=0.0, vmax=1.0),
        alpha=alpha,
        rasterized=True,
    )

    if np.any(valid):
        levels = np.quantile(density_t[valid], [0.45, 0.7, 0.88])
        levels = np.unique(levels[levels > 0])
        if len(levels) > 0:
            ax.contour(
                0.5 * (xbins[:-1] + xbins[1:]),
                0.5 * (ybins[:-1] + ybins[1:]),
                density_t,
                levels=levels,
                colors=DENSITY_LINE_COLOR,
                linewidths=1.0,
                alpha=0.65,
            )

    if show_points:
        point_budget = 2000
        if x.shape[0] > point_budget:
            step = max(1, x.shape[0] // point_budget)
            idx = np.arange(0, x.shape[0], step)
        else:
            idx = np.arange(x.shape[0])
        ax.scatter(
            x[idx][~success[idx]],
            y[idx][~success[idx]],
            s=10,
            c=FAIL_POINT_COLOR,
            alpha=0.12,
            linewidths=0,
            label="failed init",
        )
        ax.scatter(
            x[idx][success[idx]],
            y[idx][success[idx]],
            s=12,
            c=SUCCESS_POINT_COLOR,
            alpha=0.20,
            linewidths=0,
            label="successful init",
        )

    ax.set_title("Success Landscape", loc="left")
    ax.set_xlabel("initial x")
    ax.set_ylabel("initial y")
    ax.set_aspect("equal", adjustable="box")
    _style_axis(ax)
    return mesh


def _draw_density_panel(ax, total, success_hist, xbins, ybins):
    total_t = total.T
    success_t = success_hist.T
    mesh = ax.pcolormesh(
        xbins,
        ybins,
        np.ma.masked_less_equal(total_t, 0.0),
        shading="auto",
        cmap="Greys",
        norm=colors.PowerNorm(gamma=0.55, vmin=0.0, vmax=max(float(np.nanmax(total_t)), 1.0)),
        rasterized=True,
    )
    if np.nanmax(success_t) > 0:
        ax.contour(
            0.5 * (xbins[:-1] + xbins[1:]),
            0.5 * (ybins[:-1] + ybins[1:]),
            success_t,
            levels=4,
            colors=SUCCESS_POINT_COLOR,
            linewidths=1.0,
            alpha=0.7,
        )
    ax.set_title("Sample Density", loc="left")
    ax.set_xlabel("initial x")
    ax.set_ylabel("initial y")
    ax.set_aspect("equal", adjustable="box")
    _style_axis(ax)
    return mesh


def plot_one_cube(
    raw_metrics,
    cube_name,
    output_dir,
    bins,
    x_range,
    y_range,
    smooth_sigma,
    min_density_ratio,
    cmap,
    show_points,
):
    success = raw_metrics["success_once"].detach().cpu().numpy().astype(bool).reshape(-1)
    positions = _get_cube_positions(raw_metrics, cube_name).detach().cpu().numpy()
    if positions.shape[0] != success.shape[0]:
        raise ValueError(
            f"{cube_name} position count {positions.shape[0]} does not match "
            f"success count {success.shape[0]}."
        )

    x = positions[:, 0]
    y = positions[:, 1]
    x_range = x_range or _infer_range(x)
    y_range = y_range or _infer_range(y)
    xbins = np.linspace(x_range[0], x_range[1], bins + 1)
    ybins = np.linspace(y_range[0], y_range[1], bins + 1)

    total, _, _ = np.histogram2d(x, y, bins=[xbins, ybins])
    succ, _, _ = np.histogram2d(x[success], y[success], bins=[xbins, ybins])
    smooth_total = _gaussian_filter2d(total, smooth_sigma)
    smooth_success = _gaussian_filter2d(succ, smooth_sigma)
    density_threshold = float(np.nanmax(smooth_total)) * min_density_ratio
    rate = np.divide(
        smooth_success,
        smooth_total,
        out=np.full_like(smooth_success, np.nan),
        where=smooth_total > density_threshold,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cube_name}_success_heatmap"
    np.savez(
        output_dir / f"{stem}.npz",
        xbins=xbins,
        ybins=ybins,
        total=total,
        success=succ,
        success_rate=rate,
        smooth_total=smooth_total,
        smooth_success=smooth_success,
        positions=positions,
        success_once=success,
    )

    _setup_plot_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.6, 5.8),
        gridspec_kw={"width_ratios": [1.0, 0.9]},
    )
    fig.suptitle(f"{cube_name} initial-position analysis", x=0.075, y=0.98, ha="left")
    fig.text(
        0.075,
        0.93,
        (
            f"{success.size} episodes  |  "
            f"success rate {success.mean():.1%}  |  "
            f"bins {bins}x{bins}  |  "
            f"smoothing σ={smooth_sigma:g}"
        ),
        fontsize=10.5,
        color="#5c5348",
    )

    rate_mesh = _draw_rate_panel(
        axes[0], rate, smooth_total, xbins, ybins, x, y, success, cmap, show_points
    )
    density_mesh = _draw_density_panel(axes[1], total, succ, xbins, ybins)

    rate_cbar = fig.colorbar(rate_mesh, ax=axes[0], fraction=0.047, pad=0.03)
    rate_cbar.set_label("success rate")
    rate_cbar.outline.set_edgecolor(GRID_COLOR)
    density_cbar = fig.colorbar(density_mesh, ax=axes[1], fraction=0.047, pad=0.03)
    density_cbar.set_label("sample count")
    density_cbar.outline.set_edgecolor(GRID_COLOR)

    if show_points:
        axes[0].legend(
            loc="upper left",
            frameon=True,
            facecolor=AX_BG,
            edgecolor=GRID_COLOR,
            fontsize=9.5,
        )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.9])
    plt.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_path", type=Path)
    parser.add_argument("--cube", choices=["cube_1", "cube_2", "cube_3", "all"], default="cube_2")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bins", type=int, default=160)
    parser.add_argument("--x-range", type=_parse_range, default=None, help="Optional range formatted as min,max")
    parser.add_argument("--y-range", type=_parse_range, default=None, help="Optional range formatted as min,max")
    parser.add_argument("--smooth-sigma", type=float, default=5.0, help="Gaussian smoothing sigma in heatmap bins")
    parser.add_argument("--min-density-ratio", type=float, default=0.01, help="Hide cells with smoothed sample density below this fraction of the maximum")
    parser.add_argument("--cmap", default=DEFAULT_CMAP, help="Matplotlib colormap name")
    parser.add_argument("--show-points", action="store_true", help="Overlay sampled initial positions")
    args = parser.parse_args()

    payload = torch.load(args.metrics_path, map_location="cpu", weights_only=False)
    raw_metrics = payload["raw_metrics"] if "raw_metrics" in payload else payload
    if "success_once" not in raw_metrics:
        raise KeyError("Cannot find success_once in eval metrics.")

    output_dir = args.output_dir or (args.metrics_path.parent / "cube_heatmaps")
    cube_names = ["cube_1", "cube_2", "cube_3"] if args.cube == "all" else [args.cube]
    for cube_name in cube_names:
        plot_one_cube(
            raw_metrics=raw_metrics,
            cube_name=cube_name,
            output_dir=output_dir,
            bins=args.bins,
            x_range=args.x_range,
            y_range=args.y_range,
            smooth_sigma=args.smooth_sigma,
            min_density_ratio=args.min_density_ratio,
            cmap=args.cmap,
            show_points=args.show_points,
        )
        print(f"Saved {cube_name} heatmap to {output_dir}")


if __name__ == "__main__":
    main()
