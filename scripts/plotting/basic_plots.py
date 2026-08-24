#!/usr/bin/env python3
"""Basic plotting entry points for ECM data.

This module keeps the workflow that the older 2024 structure used, but makes it
work in the newer repository layout.

The high-level process is:
1. pick a core from metadata,
2. find all sections for that core,
3. for each section, locate the relevant AC/DC files in clean/ if available and
   otherwise fall back to not-cleaned/,
4. apply the normal ECM processing steps (rem_ends, smooth, norm), and
5. save a plot per section in a core-specific figures directory.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CORE_SCRIPTS = ROOT / "scripts" / "core_scripts"
if str(CORE_SCRIPTS) not in sys.path:
    sys.path.append(str(CORE_SCRIPTS))
from ECMclass import ECM  # noqa: E402

DEFAULT_DATA_ROOT = ROOT / "data"
DEFAULT_FIGURES_ROOT = ROOT / "figures"


def section_sort_key(section_value):
    """Return a stable sort key for section labels like 101, 15_1, or 268_5."""
    text = str(section_value).strip()
    match = re.search(r"(\d+)", text)
    if match is None:
        return (-1, text)
    return (int(match.group(1)), text)


def section_rows_for_core(meta, core_name):
    core_name = str(core_name).strip()
    subset = meta.loc[meta["core"].astype(str).str.upper() == core_name.upper()].copy()
    if subset.empty:
        raise ValueError(f"No metadata rows found for core {core_name!r}.")
    subset = subset.drop_duplicates(subset=["section"], keep="first")
    subset = subset.sort_values("section", key=lambda s: s.map(section_sort_key))
    return subset


def choose_section_layout(section_rows):
    """Return the plotting orientation for a section.

    The old repository used three common geometries: left/top (t/l), right/top
    (tr/r), and wide single-face layouts (w1/w2). The metadata is inspected to
    choose the correct orientation automatically.
    """
    face_set = {str(face).strip().lower() for face in section_rows["face"].dropna()}

    if face_set.intersection({"tr", "r"}):
        return "tr_r"
    if face_set.intersection({"tl", "l"}):
        return "tl_l"
    if face_set.intersection({"t"}):
        return "tr_r" if "r" in face_set or "tr" in face_set else "tl_l"
    if face_set.intersection({"w1", "w2", "half", "o", "o_lores"}):
        return "w1"
    return "default"


def resolve_data_file(core_name, section_value, face_value, ac_or_dc, path_to_data):
    """Resolve the best available file path for a core/section/face/acdc combo.

    Preference is clean data first, then not-cleaned. File names use the pattern
    {core}-{section}-{face}-{ACorDC}.csv.
    """
    section_str = str(section_value).strip()
    face_str = str(face_value).strip()
    ac_or_dc = str(ac_or_dc).strip().upper()
    file_name = f"{core_name}-{section_str}-{face_str}-{ac_or_dc}.csv"
    base = Path(path_to_data)

    for data_root_name in ["clean", "not-cleaned"]:
        check_path = base / data_root_name / core_name / file_name
        if check_path.exists():
            return check_path, data_root_name

    raise FileNotFoundError(
        f"Could not find file for {core_name}, section {section_str}, face {face_str}, {ac_or_dc} "
        f"under {base}."
    )


def prepare_ecm(data_root, metadata_path, row):
    """Create an ECM object and apply the standard processing steps."""
    core_name = str(row["core"]).strip()
    section_value = row["section"]
    face_value = row["face"]
    ac_or_dc = row["ACorDC"]

    data_root = str(data_root).rstrip("/") + "/"
    metadata_path = str(metadata_path)
    ecm = ECM(core_name, section_value, face_value, ac_or_dc, data_root, metadata_path)

    # Leave the defaults off unless the user requested them explicitly.
    # The class methods are safe to call on an object that has not yet been smoothed.
    try:
        ecm.rem_ends(0)
    except Exception:
        pass
    return ecm


def add_flag_markers(ax, ecm, raw_df, source_kind):
    """Overlay the legacy button or clean semantic flags on a panel."""
    pairs = list(zip(raw_df["Y_dimension(mm)"].astype(float), raw_df["True_depth(m)"].astype(float)))
    flag_lookup = {pair: idx for idx, pair in enumerate(pairs)}

    def aligned_mask(values):
        values = np.asarray(values, dtype=int)
        mask = []
        for y_val, depth_val in zip(ecm.y, ecm.depth):
            pair = (float(y_val), float(depth_val))
            idx = flag_lookup.get(pair)
            mask.append(bool(values[idx]) if idx is not None else False)
        return np.asarray(mask, dtype=bool)

    if source_kind == "clean":
        flag_specs = [
            ("surface_imperfection", "black"),
            ("tephra_or_layer", "red"),
            ("rock_or_debris", "green"),
        ]
        for flag_name, color in flag_specs:
            if flag_name not in raw_df.columns:
                continue
            flagged = aligned_mask(raw_df[flag_name].to_numpy().astype(int))
            if not np.any(flagged):
                continue
            ax.scatter(
                ecm.meas[flagged],
                ecm.depth[flagged],
                s=12,
                color=color,
                alpha=0.8,
                linewidths=0,
                zorder=5,
            )
        return

    button_source = raw_df["Button"].to_numpy() if "Button" in raw_df.columns else raw_df["button"].to_numpy()
    button_mask = aligned_mask(button_source.astype(int))
    if np.any(button_mask):
        ax.scatter(
            ecm.meas[button_mask],
            ecm.depth[button_mask],
            s=12,
            color="black",
            alpha=0.8,
            linewidths=0,
            zorder=5,
        )


def plotquarter(ax, y_vec, ycor, depth, meas, button, rescale, downsample=0.0):
    """Old-structure 2D rectangle plotting: each track is a band of colored rectangles.

    downsample is interpreted as the desired depth spacing in meters, matching the
    original ALHIC2416 notebook behavior (e.g. 0.002 for 2 mm depth bins).
    """
    width = y_vec[1] - y_vec[0]

    for y in y_vec:
        idx = ycor == y
        tmeas = meas[idx]
        tbut = button[idx]
        td = depth[idx]

        if downsample and downsample > 0:
            int_lo = round(float(np.min(td)), 2)
            int_hi = round(float(np.max(td)), 2)
            depth_interp = np.linspace(int_lo, int_hi, int((int_hi - int_lo) / downsample) + 1)
            meas_interp = np.interp(depth_interp, np.flip(td), np.flip(tmeas))
            but_interp = np.interp(depth_interp, np.flip(td), np.flip(tbut))
            td = depth_interp
            tmeas = meas_interp
            tbut = np.round(but_interp)

        for i in range(len(tmeas) - 1):
            if tbut[i] == 0:
                color = matplotlib.colormaps["Spectral"](rescale(tmeas[i]))
            else:
                color = "k"
            ax.add_patch(
                Rectangle(
                    (y - (width - 0.2) / 2, td[i]),
                    (width - 0.2),
                    td[i + 1] - td[i],
                    facecolor=color,
                    edgecolor="none",
                )
            )


def plot_section(core_name, section_value, meta, path_to_data, path_to_figures, window=10, downsample=1, normalize=True, trim_ends=0):
    """Plot all files for a single section using the old 2024 layout design."""
    section_df = meta.loc[
        (meta["core"].astype(str).str.upper() == str(core_name).strip().upper())
        & (meta["section"].astype(str).str.strip() == str(section_value).strip())
    ].copy()
    if section_df.empty:
        raise ValueError(f"No metadata rows for core {core_name!r}, section {section_value!r}.")

    layout = choose_section_layout(section_df)
    data_root = Path(path_to_data)
    plot_dir = Path(path_to_figures) / str(core_name) / f"{core_name}_{layout}_window{window}_downsample{downsample}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    faces_by_acdc = {
        "AC": {"t": None, "l": None, "r": None, "tr": None, "w1": None},
        "DC": {"t": None, "l": None, "r": None, "tr": None, "w1": None},
    }

    for _, row in section_df.iterrows():
        face_key = str(row["face"]).strip().lower()
        acdc_key = str(row["ACorDC"]).strip().upper()
        file_path, source_kind = resolve_data_file(core_name, row["section"], row["face"], row["ACorDC"], path_to_data)
        ecm = prepare_ecm(str(data_root / source_kind), str(data_root / "metadata.csv"), row.to_dict())

        if trim_ends and trim_ends > 0:
            try:
                ecm.rem_ends(float(trim_ends))
            except Exception:
                pass

        if window and window > 0:
            try:
                ecm.smooth(float(window))
            except Exception:
                pass

        if normalize:
            try:
                ecm.norm_all()
            except Exception:
                pass

        faces_by_acdc[acdc_key].setdefault(face_key, {})
        faces_by_acdc[acdc_key][face_key] = {"ecm": ecm, "source_kind": source_kind, "file_path": file_path}

    if layout == "tr_r":
        face_order = [("AC", "r"), ("AC", "t"), ("DC", "r"), ("DC", "t")]
        fig, ax = plt.subplots(1, 5, gridspec_kw={"width_ratios": [2, 3, 2, 2, 3]}, figsize=(9, 6), dpi=200)
        title_order = ["AC - Right", "AC - Top", "", "DC - Right", "DC - Top"]
        axis_xlim = [(60, 0), (0, 120), None, (60, 0), (0, 120)]
    elif layout == "tl_l":
        face_order = [("AC", "l"), ("AC", "t"), ("DC", "l"), ("DC", "t")]
        fig, ax = plt.subplots(1, 5, gridspec_kw={"width_ratios": [2, 3, 2, 2, 3]}, figsize=(9, 6), dpi=200)
        title_order = ["AC - Left", "AC - Top", "", "DC - Left", "DC - Top"]
        axis_xlim = [(120, 0), (0, 120), None, (120, 0), (0, 120)]
    else:
        face_order = [("AC", "w1"), ("DC", "w1")]
        fig, ax = plt.subplots(1, 2, figsize=(7, 5), dpi=200)
        title_order = ["AC", "DC"]
        axis_xlim = [None, None]

    if layout == "w1":
        all_depths = []
        all_meas = []
        for acdc, face in face_order:
            item = faces_by_acdc[acdc].get(face)
            if item is None:
                continue
            ecm = item["ecm"]
            all_depths.extend(ecm.depth_s if hasattr(ecm, "depth_s") else ecm.depth)
            all_meas.extend(ecm.meas_s if hasattr(ecm, "meas_s") else ecm.meas)

        if all_depths:
            dmin = min(all_depths)
            dmax = max(all_depths)
        else:
            dmin = 0
            dmax = 1

        for panel_idx, (acdc, face) in enumerate(face_order):
            item = faces_by_acdc[acdc].get(face)
            if item is None:
                ax[panel_idx].axis("off")
                continue
            ecm = item["ecm"]
            yvec = np.unique(ecm.y_s if hasattr(ecm, "y_s") else ecm.y)
            yall = ecm.y_s if hasattr(ecm, "y_s") else ecm.y
            if acdc == "AC":
                vmin, vmax = np.percentile(np.concatenate([ecm.meas_s, ecm.meas]), 5), np.percentile(np.concatenate([ecm.meas_s, ecm.meas]), 95)
            else:
                vmin, vmax = np.percentile(np.concatenate([ecm.meas_s, ecm.meas]), 5), np.percentile(np.concatenate([ecm.meas_s, ecm.meas]), 95)
            rescale = lambda k, lo=vmin, hi=vmax: 0 if hi == lo else (k - lo) / (hi - lo)
            plotquarter(ax[panel_idx], yvec, yall, ecm.depth_s if hasattr(ecm, "depth_s") else ecm.depth, ecm.meas_s if hasattr(ecm, "meas_s") else ecm.meas, ecm.button_s if hasattr(ecm, "button_s") else ecm.button, rescale, downsample=downsample)
            add_flag_markers(ax[panel_idx], ecm, pd.read_csv(item["file_path"]), item["source_kind"])
            ax[panel_idx].set_title(title_order[panel_idx])
            ax[panel_idx].set_ylabel("Depth (m)")
            ax[panel_idx].set_xlabel("Distance From Center (mm)", fontsize=6)
            ax[panel_idx].set_ylim([dmax, dmin])
        fig.suptitle(f"{core_name} - {section_value} - {str(window)} mm smooth")
        fig.tight_layout(); plt.subplots_adjust(wspace=0)
        output_path = plot_dir / f"{core_name}-{section_value}-{layout}.png"
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        return output_path

    all_depths = []
    all_meas = []
    for acdc, face in face_order:
        item = faces_by_acdc[acdc].get(face)
        if item is None:
            continue
        ecm = item["ecm"]
        all_depths.extend(ecm.depth_s if hasattr(ecm, "depth_s") else ecm.depth)
        all_meas.extend(ecm.meas_s if hasattr(ecm, "meas_s") else ecm.meas)

    dmin = min(all_depths)
    dmax = max(all_depths)
    ac_vals = []
    dc_vals = []
    for acdc, face in face_order:
        item = faces_by_acdc[acdc].get(face)
        if item is None:
            continue
        ecm = item["ecm"]
        vals = ecm.meas_s if hasattr(ecm, "meas_s") else ecm.meas
        if acdc == "AC":
            ac_vals.extend(vals)
        else:
            dc_vals.extend(vals)

    ACpltmin = np.percentile(ac_vals, 5) if ac_vals else np.nan
    ACpltmax = np.percentile(ac_vals, 95) if ac_vals else np.nan
    DCpltmin = np.percentile(dc_vals, 5) if dc_vals else np.nan
    DCpltmax = np.percentile(dc_vals, 95) if dc_vals else np.nan

    fig.suptitle(f"{core_name} - {section_value} - {str(window)} mm smooth")
    ax[2].axis("off")

    for panel_idx, (acdc, face) in enumerate(face_order):
        item = faces_by_acdc[acdc].get(face)
        if item is None:
            ax[panel_idx].axis("off")
            continue

        ecm = item["ecm"]
        if face in {"r", "tr"}:
            yall = ecm.y_s - ecm.y_left if hasattr(ecm, "y_left") else ecm.y - np.min(ecm.y)
            yvec = np.unique(ecm.y_s if hasattr(ecm, "y_s") else ecm.y) - (ecm.y_left if hasattr(ecm, "y_left") else np.min(ecm.y))
        else:
            yall = ecm.y_right - ecm.y_s if hasattr(ecm, "y_right") else np.max(ecm.y) - ecm.y
            yvec = (ecm.y_right if hasattr(ecm, "y_right") else np.max(ecm.y)) - np.unique(ecm.y_s if hasattr(ecm, "y_s") else ecm.y)

        if acdc == "AC":
            rescale = lambda k, lo=ACpltmin, hi=ACpltmax: 0 if hi == lo else (k - lo) / (hi - lo)
        else:
            rescale = lambda k, lo=DCpltmin, hi=DCpltmax: 0 if hi == lo else (k - lo) / (hi - lo)

        plotquarter(ax[panel_idx], yvec, yall, ecm.depth_s if hasattr(ecm, "depth_s") else ecm.depth, ecm.meas_s if hasattr(ecm, "meas_s") else ecm.meas, ecm.button_s if hasattr(ecm, "button_s") else ecm.button, rescale, downsample=downsample)
        raw_df = pd.read_csv(item["file_path"])
        add_flag_markers(ax[panel_idx], ecm, raw_df, item["source_kind"])
        ax[panel_idx].set_ylabel("Depth (m)")
        ax[panel_idx].set_xlabel("Distance From Center (mm)", fontsize=6)
        ax[panel_idx].set_ylim([dmax, dmin])
        ax[panel_idx].set_title(title_order[panel_idx])
        if axis_xlim[panel_idx] is not None:
            ax[panel_idx].set_xlim(axis_xlim[panel_idx])

    fig.tight_layout(); plt.subplots_adjust(wspace=0)

    ACcbar_ax = fig.add_axes([0.07, -0.05, 0.35, 0.05])
    DCcbar_ax = fig.add_axes([0.58, -0.05, 0.35, 0.05])
    ACnorm = matplotlib.colors.Normalize(vmin=ACpltmin, vmax=ACpltmax)
    DCnorm = matplotlib.colors.Normalize(vmin=DCpltmin, vmax=DCpltmax)
    if np.isfinite(ACpltmin) and np.isfinite(ACpltmax):
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=ACnorm, cmap=matplotlib.colormaps["Spectral"]), cax=ACcbar_ax, orientation="horizontal", label="Current (amps)")
    if np.isfinite(DCpltmin) and np.isfinite(DCpltmax):
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=DCnorm, cmap=matplotlib.colormaps["Spectral"]), cax=DCcbar_ax, orientation="horizontal", label="Current (amps)")

    output_path = plot_dir / f"{core_name}-{section_value}-{layout}.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return output_path


def master_plot(core_name, path_to_data="../../data/", path_to_figures="../../figures/", metadata_file="metadata.csv", window=10, downsample=1, normalize=True, trim_ends=0):
    """Create and save all plots for a core.

    Parameters
    ----------
    core_name : str
        Core identifier to plot, e.g., 'ALHIC2201'.
    path_to_data : str or Path
        Root directory containing metadata.csv and either clean/ or not-cleaned/.
    path_to_figures : str or Path
        Root directory for figure output.
    metadata_file : str
        Metadata filename to read.
    window : float
        Smoothing window in mm.
    downsample : int
        Plot downsampling factor. Set to 1 for full resolution or greater than 1 to
        speed up plotting.
    normalize : bool
        Whether to normalize the meas traces after smoothing.
    trim_ends : float
        Number of mm to trim from the ends before plotting.
    """
    data_root = Path(path_to_data)
    metadata_path = data_root / metadata_file
    meta = pd.read_csv(metadata_path)
    sections = section_rows_for_core(meta, core_name)

    output_paths = []
    total_sections = len(sections["section"].drop_duplicates())
    for idx, section_value in enumerate(sections["section"].drop_duplicates(), start=1):
        print(f"Plotting {core_name} section {section_value}, {idx} of {total_sections}")
        output = plot_section(
            core_name,
            section_value,
            meta,
            str(data_root),
            str(path_to_figures),
            window=window,
            downsample=downsample,
            normalize=normalize,
            trim_ends=trim_ends,
        )
        output_paths.append(output)

    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Generate basic ECM plots for a given core.")
    parser.add_argument("core", help="Core name, e.g. ALHIC2201")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Path to the data directory")
    parser.add_argument("--figures-root", default=str(DEFAULT_FIGURES_ROOT), help="Root directory for figures")
    parser.add_argument("--metadata-file", default="metadata.csv", help="Metadata filename")
    parser.add_argument("--window", type=float, default=10.0, help="Smoothing window in mm")
    parser.add_argument("--downsample", type=float, default=0.002, help="Depth spacing in meters for downsampling (e.g. 0.002 = 2 mm, matching ALHIC2416)")
    parser.add_argument("--trim-ends", type=float, default=0.0, help="Number of mm to trim from the ends")
    parser.add_argument("--no-normalize", action="store_true", help="Skip normalization step")
    args = parser.parse_args()

    master_plot(
        args.core,
        path_to_data=args.data_root,
        path_to_figures=args.figures_root,
        metadata_file=args.metadata_file,
        window=args.window,
        downsample=args.downsample,
        normalize=not args.no_normalize,
        trim_ends=args.trim_ends,
    )


if __name__ == "__main__":
    main()
