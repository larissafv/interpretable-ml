"""Plotting utilities for interpretability artifacts.

This module centralizes Matplotlib-based helpers to visualize multivariate
time series, differences between instances, and heatmap-style importances.
All plots are saved into the directory configured by `common.conf.PLOT_PATH`.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .conf import PLOT_PATH


def plot_timeseries_with_highlights(
    time_series: np.ndarray,
    highlights: np.ndarray,
    plot_name: Path,
    variate_labels: list[str] = None,
    title: str | None = None,
) -> None:
    """Plot multivariate time series with selected points highlighted.

    The function overlays scatter markers on timesteps where the
    binary mask `highlights` equals 1 for each variate.

    Args:
        time_series: Array of shape (n_variates, n_timesteps).
        highlights: Binary mask with the same shape as `time_series`.
        plot_name: File name (relative) for the output PNG.
        variate_labels: Optional per-variate labels for y axes.
        title: Optional title to display atop the figure.
    """
    n_variates, n_timesteps = time_series.shape
    if variate_labels is None:
        variate_labels = [f"Var {i + 1}" for i in range(n_variates)]

    fig, axes = plt.subplots(n_variates, 1, figsize=(12, 2.5 * n_variates), sharex=True)
    if n_variates == 1:
        axes = [axes]

    for i in range(n_variates):
        ax = axes[i]
        ax.plot(time_series[i], color="black", linewidth=1)
        highlight_indices = np.where(highlights[i] == 1)[0]
        ax.scatter(
            highlight_indices,
            time_series[i, highlight_indices],
            color="orange",
            s=20,
            label="Modified Points",
        )
        ax.set_ylabel(variate_labels[i])
        ax.set_xlim(0, n_timesteps)
        # Match vertical scale to the heatmap-based plots which use
        # extent=[0, n_timesteps, np.min(time_series), np.max(time_series)].
        # This ensures consistent y-axis limits across different plot types.
        ax.set_ylim(np.min(time_series), np.max(time_series))
    axes[-1].set_xlabel("Time")
    if title:
        fig.suptitle(title)
        plt.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        plt.tight_layout()
    plt.savefig(PLOT_PATH / plot_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_highlighted_slices(
    time_series: np.ndarray,
    n_slices: int,
    top_slices: list,
    plot_name: Path,
    variate_labels: list[str] = None,
    title: str | None = None,
) -> None:
    """Plot multivariate time series with highlighted slice intervals.

    Each variate is divided into `n_slices` intervals; those whose
    indices are present in `top_slices[i]` are shaded.

    Args:
        time_series: Array of shape (n_variates, n_timesteps).
        n_slices: Number of equal partitions per variate.
        top_slices: List of lists of slice indices to highlight per variate.
        plot_name: File name (relative) for the output PNG.
        variate_labels: Optional labels for y axes.
        title: Optional title to display atop the figure.
    """
    n_variates, n_timesteps = time_series.shape
    slice_len = n_timesteps // n_slices
    if variate_labels is None:
        variate_labels = [f"Var {i + 1}" for i in range(n_variates)]

    fig, axes = plt.subplots(n_variates, 1, figsize=(12, 2.5 * n_variates), sharex=True)
    if n_variates == 1:
        axes = [axes]

    for i in range(n_variates):
        ax = axes[i]
        ax.plot(time_series[i], color="black", linewidth=1)
        for s in range(n_slices):
            start = s * slice_len
            end = (s + 1) * slice_len if s < n_slices - 1 else n_timesteps
            if s in top_slices[i]:
                ax.axvspan(start, end, color="orange", alpha=0.4)
            # Draw slice boundaries
            ax.axvline(start, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_ylabel(variate_labels[i])
        ax.set_xlim(0, n_timesteps)
        # Match vertical scale to the heatmap-based plots so all visualizations
        # share a consistent y-axis (global min/max across the whole time_series)
        ax.set_ylim(np.min(time_series), np.max(time_series))
    axes[-1].set_xlabel("Time")
    if title:
        fig.suptitle(title)
        plt.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        plt.tight_layout()
    plt.savefig(PLOT_PATH / plot_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_timeseries_differences(
    ts1: np.ndarray,
    ts2: np.ndarray,
    plot_name: Path,
    variate_labels: list[str] = None,
    title: str | None = None,
) -> None:
    """Plot two time series and shade regions where they differ.

    Args:
        ts1: First time series (n_variates, n_timesteps).
        ts2: Second time series with the same shape as ts1.
        plot_name: File name (relative) for the output PNG.
        variate_labels: Optional labels for y axes.
        title: Optional title to display atop the figure.
    """
    n_variates, n_timesteps = ts1.shape
    if variate_labels is None:
        variate_labels = [f"Var {i + 1}" for i in range(n_variates)]

    fig, axes = plt.subplots(n_variates, 1, figsize=(12, 2.5 * n_variates), sharex=True)
    if n_variates == 1:
        axes = [axes]

    for i in range(n_variates):
        ax = axes[i]
        ax.plot(ts1[i], label="Original ECG", color="blue", alpha=0.9)
        ax.plot(ts2[i], label="Perturbed ECG", color="red", alpha=0.6)
        # Highlight differences
        diff_mask = ts1[i] != ts2[i]
        if np.any(diff_mask):
            # Find contiguous regions of difference
            diff_indices = np.where(diff_mask)[0]
            if diff_indices.size > 0:
                # Group contiguous indices
                groups = np.split(diff_indices, np.where(np.diff(diff_indices) != 1)[0] + 1)
                for group in groups:
                    ax.axvspan(group[0], group[-1] + 1, color="orange", alpha=0.3)
        ax.set_ylabel(variate_labels[i])
        ax.legend(loc="upper right")
        ax.set_xlim(0, n_timesteps)
        # Use combined min/max across both series so difference plots align
        global_min = min(np.min(ts1), np.min(ts2))
        global_max = max(np.max(ts1), np.max(ts2))
        ax.set_ylim(global_min, global_max)
    axes[-1].set_xlabel("Time")
    if title:
        fig.suptitle(title)
        plt.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        plt.tight_layout()
    plt.savefig(PLOT_PATH / plot_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_timeseries_with_heatmap(
    time_series: np.ndarray,
    heatmap: np.ndarray,
    plot_name: Path,
    variate_labels: list[str] = None,
    cmap: str = "twilight_shifted",
    colorbar_pad: float = 0.025,
    title: str | None = None,
) -> None:
    """Overlay a 1D heatmap on each variate of a multivariate time series.

    The heatmap is rendered with imshow as a single-row image whose vertical
    extent matches the value range of each variate, enabling a readable overlay.

    Args:
        time_series: Array of shape (n_variates, n_timesteps).
        heatmap: Importance scores for each variate, shape (n_variates, n_timesteps).
        plot_name: File name (relative) for the output PNG.
        variate_labels: Optional per-variate labels; defaults to Var i.
        cmap: Matplotlib colormap name for the heatmap.
        colorbar_pad: Padding between subplots and colorbar.
        title: Optional title to display atop the figure.
    """
    n_variates, n_timesteps = time_series.shape
    if variate_labels is None:
        variate_labels = [f"Var {i + 1}" for i in range(n_variates)]

    fig, axes = plt.subplots(n_variates, 1, figsize=(12, 2.5 * n_variates), sharex=True)
    if n_variates == 1:
        axes = [axes]

    # Plot the first heatmap to get the mappable for the colorbar
    im = None
    max_heatmap = np.max(np.abs(heatmap))
    for i in range(n_variates):
        ax = axes[i]
        im = ax.imshow(
            heatmap[i][np.newaxis, :],
            aspect="auto",
            cmap=cmap,
            extent=[0, n_timesteps, np.min(time_series), np.max(time_series)],
            alpha=0.5,
            vmin=max_heatmap * -1,
            vmax=max_heatmap
        )
        ax.plot(time_series[i], color="black", linewidth=1)
        ax.set_ylabel(variate_labels[i])
        ax.set_xlim(0, n_timesteps)
        # Ensure explicit y-limits (same as extent) so other plot types match
        ax.set_ylim(np.min(time_series), np.max(time_series))

        fig.colorbar(
            im,
            ax=ax,
            pad=colorbar_pad,
            label="Heatmap value",
        )

    if title:
        fig.suptitle(title)
        plt.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        plt.tight_layout()
    plt.savefig(PLOT_PATH / plot_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_two_timeseries_with_heatmap(
    ts1: np.ndarray,
    ts2: np.ndarray,
    heatmap: np.ndarray,
    plot_name: Path,
    variate_labels: list[str] = None,
    cmap: str = "PiYG",
    colorbar_pad: float = 0.025,
    title: str | None = None,
) -> None:
    """Overlay a 1D heatmap on each variate and plot two time series per variate.

    This function is analogous to ``plot_timeseries_with_heatmap`` but plots
    two series (e.g. original and perturbed) for each variate. The heatmap is
    rendered with ``imshow`` as a single-row image whose vertical extent
    matches the combined value range of the two series so the overlay is
    readable and consistent across variates.

    Args:
        ts1: First time series array of shape (n_variates, n_timesteps).
        ts2: Second time series array with the same shape as ``ts1``.
        heatmap: Importance scores for each variate, shape (n_variates, n_timesteps).
        plot_name: File name (relative) for the output PNG.
        variate_labels: Optional per-variate labels; defaults to Var i.
        cmap: Matplotlib colormap name for the heatmap.
        colorbar_pad: Padding between subplots and colorbar.
        title: Optional title to display atop the figure.
    """
    n_variates, n_timesteps = ts1.shape
    if variate_labels is None:
        variate_labels = [f"Var {i + 1}" for i in range(n_variates)]

    fig, axes = plt.subplots(n_variates, 1, figsize=(12, 2.5 * n_variates), sharex=True)
    if n_variates == 1:
        axes = [axes]

    max_heatmap = np.max(np.abs(heatmap))
    # Combined min/max across both series so overlay aligns with both lines
    global_min = min(np.min(ts1), np.min(ts2))
    global_max = max(np.max(ts1), np.max(ts2))

    for i in range(n_variates):
        ax = axes[i]
        im = ax.imshow(
            heatmap[i][np.newaxis, :],
            aspect="auto",
            cmap=cmap,
            extent=[0, n_timesteps, global_min, global_max],
            alpha=0.5,
            vmin=max_heatmap * -1,
            vmax=max_heatmap,
        )
        # Plot the two series
        ax.plot(ts1[i], color="black", linewidth=1, label="Original Series")
        ax.plot(ts2[i], color="red", linewidth=1, alpha=0.3, label="Perturbed Series")
        ax.set_ylabel(variate_labels[i])
        ax.set_xlim(0, n_timesteps)
        ax.set_ylim(global_min, global_max)

        fig.colorbar(
            im,
            ax=ax,
            pad=colorbar_pad,
            label="Heatmap value",
        )
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time")
    if title:
        fig.suptitle(title)
        plt.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        plt.tight_layout()
    plt.savefig(PLOT_PATH / plot_name, dpi=300, bbox_inches="tight")
    plt.close(fig)
