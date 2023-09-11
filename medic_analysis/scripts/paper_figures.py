"""Main script for generating paper figures."""
import os
import argparse
import json
from pathlib import Path
from PIL import Image
import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import ttest_rel, pearsonr
import pandas as pd
import seaborn as sns
from skimage.exposure import equalize_hist, equalize_adapthist
from medic_analysis.common import data_plotter, render_dynamic_figure, hz_limits_to_mm
from . import (
    DATA_DIR,
    FIGURES_DIR,
    MM_TO_INCHES,
)
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps


# Set global seaborn figure settings
GLOBAL_SETTINGS = {
    "font": "Satoshi",
    "font_scale": 1,
    "palette": "pastel",
    "style": "white",
    "rc": {
        "figure.dpi": 150,
        "figure.titlesize": 7,
        "font.size": 7,
        "axes.titlesize": 6,
        "axes.titlepad": 0,
        "axes.labelsize": 6,
        "axes.labelpad": 0,
        "axes.linewidth": 0.5,
        "legend.title_fontsize": 6,
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "xtick.major.pad": 1,
        "xtick.major.size": 1,
        "ytick.labelsize": 6,
        "ytick.major.pad": 1,
        "ytick.major.size": 1,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
    },
}
sns.set_theme(**GLOBAL_SETTINGS)
LOWER_FONT_SIZE = 5


# Default paths for data
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
# Head position figure
FIGURE1_DATA = "/home/usr/vana/GMT2/Andrew/HEADPOSITIONSUSTEST"
# Concatenated head position figure
FIGURE2_DATA = "/home/usr/vana/GMT2/Andrew/HEADPOSITIONCAT"
# Group Template Analysis
FIGURE3_DATA = str(DATA_DIR)
# Alignment and Field map Comparison
FIGURE4_DATA = str(DATA_DIR)
# Spotlight Analysis figure
FIGURE5_DATA = str(DATA_DIR)
# Alignment metrics
FIGURE6_DATA = str(DATA_DIR / "alignment_metrics.csv")
# Field map metrics
FIGURE7_DATA = str(DATA_DIR)
# tSNR figure
FIGURE10_DATA = str(DATA_DIR / "tsnr.csv")
# dynamic field map videos
FIGURE100_DATA = "/home/usr/vana/GMT2/Andrew/HEADPOSITIONSUSTEST/derivatives"
AA_DATA_DIR = Path("/data/Daenerys/ASD_ADHD/NP1173/derivatives/me_pipeline2")
WASHU_DATA_DIR = Path("/net/10.20.145.34/DOSENBACH02/GMT2/Andrew/SLICETEST/derivatives/me_pipeline")
PENN_DATA_DIR = Path("/net/10.20.145.34/DOSENBACH02/GMT2/Andrew/UPenn/derivatives/me_pipeline")
MINN_DATA_DIR = Path("/net/10.20.145.34/DOSENBACH02/GMT2/Andrew/UMinn/derivatives")
MINN_DATA_DIR2 = Path("/data/nil-bluearc/GMT/Laumann/Pilot_ME_res/BIO10001/bids/derivatives/me_pipeline")


def plot_box_plot(data, variable, label, ax):
    p = sns.color_palette("pastel")
    subdata = (
        data[[f"{variable}_medic", f"{variable}_topup"]]
        .rename(columns={f"{variable}_medic": "MEDIC", f"{variable}_topup": "TOPUP"})
        .melt(var_name=label)
    )
    sb = sns.boxplot(
        data=subdata,
        x="value",
        y=label,
        order=["MEDIC", "TOPUP"],
        ax=ax,
        fliersize=1,
        linewidth=0.5,
        palette=[p[1], p[0]],
    )
    sb.set_xlabel("")
    sb.set_ylabel(label)
    ax.tick_params(axis="x")
    ax.tick_params(axis="y")
    return sb


def draw_seed(ax, x, y, radius=30, fc="black", ec="white", linewidth=1, zorder=3):
    dcoords = ax.transAxes.transform((x, y))
    ncoords = ax.transData.inverted().transform(dcoords)
    ax.add_patch(patches.Circle(ncoords, radius, fc=fc, ec=ec, linewidth=linewidth, zorder=zorder))


def data_to_ax(ax, corrds):
    return ax.transAxes.inverted().transform(ax.transData.transform(corrds))


def draw_arrow(ax, loc1, loc2, color="red", linewidth=1, head_width=2, head_length=4):
    dloc1 = ax.transAxes.transform(loc1)
    nloc1 = ax.transData.inverted().transform(dloc1)
    dloc2 = ax.transAxes.transform(loc2)
    nloc2 = ax.transData.inverted().transform(dloc2)
    ax.add_patch(
        patches.FancyArrowPatch(
            nloc1,
            nloc2,
            arrowstyle="-|>,head_width={},head_length={}".format(head_width, head_length),
            color=color,
            linewidth=linewidth,
        )
    )


# figure 1
def head_position_fieldmap(data):
    mpl.rcParams["axes.titlesize"] = 7
    # Get the data
    output_dir = Path(data) / "derivatives"

    raw_func_path = (
        Path("/home/usr/vana/GMT2/Andrew/HEADPOSITIONCAT")
        / "sub-MSCHD02"
        / "ses-01"
        / "func"
        / "sub-MSCHD02_ses-01_task-rest_run-01_echo-1_part-mag_bold.nii.gz"
    )

    # create a list of expected labels for each run
    labels = [
        "Neutral",
        "+Z Rotation",
        "-Z Rotation",
        "+X Rotation",
        "-X Rotation",
        "+Y Rotation",
        "-Y Rotation",
        "Neutral to +Z Rotation",
        "Neutral to -Z Rotation",
        "Neutral to +X Rotation",
        "Neutral to -X Rotation",
        "Neutral to +Y Rotation",
        "Neutral to -Y Rotation",
        "Neutral to -Z Translation",
        "-Z Translation",
    ]

    # indices for run
    static_head_position_run_idx = [0, 1, 2, 3, 4, 5, 6, 14]

    # load raw_func data
    raw_func = nib.load(raw_func_path)

    # Figure 1 - Head Rotation Data
    # load field map files
    medic_fieldmaps = Path(output_dir) / "fieldmaps" / "medic_aligned"
    # load topup field map in neutral position as reference
    topup_fieldmap = nib.load(Path(output_dir) / "fieldmaps" / "topup" / "run01" / "fout.nii.gz").get_fdata()
    # load static field map runs
    static_fieldmaps = []
    for idx in static_head_position_run_idx:
        run = idx + 1
        static_fieldmaps.append(nib.load(medic_fieldmaps / f"run{run:02d}" / "fmap.nii.gz").dataobj)
    # load mask
    mask = nib.load(Path(output_dir) / "references" / "me_epi_ref_bet_mask.nii.gz").get_fdata()

    # plot range
    vlims = (-50, 50)

    # plot static field maps
    f0 = plt.figure(figsize=(180 * MM_TO_INCHES, 130 * MM_TO_INCHES), layout="constrained")

    # grid spec for head motion images
    gsm = GridSpec(
        3, 7, left=0.025, right=0.975, bottom=0.625, top=0.96, wspace=0.04, hspace=0.03, height_ratios=[110, 72, 72]
    )

    # plot movement data
    # get min max
    func_min = raw_func.dataobj[..., 0].min()
    func_max = raw_func.dataobj[..., 0].max()
    # create subplots from gridspec
    axes_list = []
    for i in range(7):
        for j in range(3):
            axes_list.append(f0.add_subplot(gsm[j, i]))
    # plot data
    data_plotter(  # f0.suptitle("Motion-dependent field map differences (Position - Neutral Position)")
        [
            raw_func.dataobj[..., 50],
            raw_func.dataobj[..., 150],
            raw_func.dataobj[..., 250],
            raw_func.dataobj[..., 350],
            raw_func.dataobj[..., 450],
            raw_func.dataobj[..., 550],
            raw_func.dataobj[..., 650],
        ],
        vmin=func_min,
        vmax=func_max,
        colormaps="gray",
        figure=f0,
        axes_list=axes_list,
    )
    sbs = axes_list
    sbs[2].set_xlabel("50", labelpad=2)
    sbs[20].set_xlabel("650", labelpad=2)
    sbs[11].set_xlabel("Frame", labelpad=2)
    sbs[0].set_title(r"$\bf{a}$    Functional MRI timeseries: High motion", pad=4, weight="normal", loc="left")

    # draw arrow line
    f0.canvas.draw()
    start = sbs[2].xaxis.label.get_window_extent()
    middle = sbs[11].xaxis.label.get_window_extent()
    end = sbs[20].xaxis.label.get_window_extent()
    # transform to figure coordinates
    start = start.transformed(f0.transFigure.inverted())
    middle = middle.transformed(f0.transFigure.inverted())
    end = end.transformed(f0.transFigure.inverted())
    # get the midpoints
    start = np.average(start.get_points(), axis=0)
    middle = np.average(middle.get_points(), axis=0)
    end = np.average(end.get_points(), axis=0)
    arrow1 = patches.FancyArrowPatch(
        start + np.array([0.01, 0]),
        middle - np.array([0.02, 0]),
        arrowstyle="-",
        color="black",
        linewidth=0.5,
    )
    arrow2 = patches.FancyArrowPatch(
        middle + np.array([0.02, 0]),
        end - np.array([0.01, 0]),
        arrowstyle="-|>,head_width=2,head_length=4",
        color="black",
        linewidth=0.5,
    )
    f0.add_artist(arrow1)
    f0.add_artist(arrow2)

    # get bounding box of last image
    bbox = sbs[-1].get_window_extent()
    # transform to figure coordinates
    bbox = bbox.transformed(f0.transFigure.inverted())

    # create a grid spec for the figure
    bottom = 0.025
    top = 0.575
    left_edge_1 = 0.09
    right_edge_2 = bbox.x1
    pad = 0.02
    width = (right_edge_2 - left_edge_1 - pad) / 2
    right_edge_1 = left_edge_1 + width
    left_edge_2 = right_edge_1 + pad
    gs0 = GridSpec(1, 1, left=0.03, right=0.09, bottom=bottom, top=top)
    gs_bar = GridSpecFromSubplotSpec(
        1, 3, wspace=0, hspace=0, width_ratios=[2, 1, 6], subplot_spec=gs0[:, :]
    )
    gs1 = GridSpec(
        3,
        3,
        left=left_edge_1,
        right=right_edge_1,
        bottom=bottom,
        top=top,
        hspace=0.025,
        wspace=0.025,
        width_ratios=[72, 110, 110],
    )
    gs2 = GridSpec(
        3,
        3,
        left=left_edge_2,
        right=right_edge_2,
        bottom=bottom,
        top=top,
        hspace=0.025,
        wspace=0.025,
        width_ratios=[72, 110, 110],
    )

    # create subplots
    cbar_ax = f0.add_subplot(gs_bar[1])
    axes_list = []
    for i in range(3):
        for j in range(3):
            axes_list.append(f0.add_subplot(gs1[i, j]))
        for j in range(3):
            axes_list.append(f0.add_subplot(gs2[i, j]))

    # plot the data
    data_plotter(
        [
            (static_fieldmaps[1][..., 0] - static_fieldmaps[0][..., 0]) * mask,
            (static_fieldmaps[2][..., 0] - static_fieldmaps[0][..., 0]) * mask,
            (static_fieldmaps[3][..., 0] - static_fieldmaps[0][..., 0]) * mask,
            (static_fieldmaps[4][..., 0] - static_fieldmaps[0][..., 0]) * mask,
            (static_fieldmaps[5][..., 0] - static_fieldmaps[0][..., 0]) * mask,
            (static_fieldmaps[6][..., 0] - static_fieldmaps[0][..., 0]) * mask,
        ],
        colorbar=True,
        colorbar_alt_range=True,
        figure=f0,
        vmin=vlims[0],
        vmax=vlims[1],
        axes_list=axes_list,
        cbar_ax=cbar_ax,
    )
    sbs = axes_list
    sbs[0].set_title(r"$\bf{b}$    " + f"{labels[1]} (15.0 deg)", pad=4, weight="normal", loc="left")
    sbs[3].set_title(r"$\bf{c}$    " + f"{labels[2]} (9.8 deg)", pad=4, weight="normal", loc="left")
    sbs[6].set_title(r"$\bf{d}$    " + f"{labels[3]} (10.6 deg)", pad=4, weight="normal", loc="left")
    sbs[9].set_title(r"$\bf{e}$    " + f"{labels[4]} (13.7 deg)", pad=4, weight="normal", loc="left")
    sbs[12].set_title(r"$\bf{f}$    " + f"{labels[5]} (10.8 deg)", pad=4, weight="normal", loc="left")
    sbs[15].set_title(r"$\bf{g}$    " + f"{labels[6]} (8.6 deg)", pad=4, weight="normal", loc="left")

    f0.savefig(FIGURES_DIR / "fieldmap_differences.png", dpi=300)
    current_dir = os.getcwd()
    os.chdir(FIGURES_DIR)
    Path("figure1.png").unlink(missing_ok=True)
    Path("figure1.png").symlink_to("fieldmap_differences.png")
    os.chdir(current_dir)
    sns.set_theme(**GLOBAL_SETTINGS)


# figure 2
def head_concatenation(data):
    # get dataset
    dataset = Path(data)

    # load medic and topup workbench screenshots
    medic_scan_path = dataset / "medic_scan.png"
    topup_scan_path = dataset / "topup_scan.png"
    truth_scan_path = dataset / "truth_scan.png"
    medic_dlpfc_path = dataset / "medic_dlpfc.png"
    topup_dlpfc_path = dataset / "topup_dlpfc.png"
    truth_dlpfc_path = dataset / "truth_dlpfc.png"
    medic_occipital_path = dataset / "medic_occipital.png"
    topup_occipital_path = dataset / "topup_occipital.png"
    truth_occipital_path = dataset / "truth_occipital.png"

    # load data
    clip1 = 50
    clip2 = 60
    clipy1 = 150
    clipy2 = 200
    medic_dlpfc = np.array(Image.open(medic_dlpfc_path))
    medic_dlpfc_left = medic_dlpfc[clipy1:-clipy2, clip1 : medic_dlpfc.shape[1] // 2 - clip2]
    medic_dlpfc_right = medic_dlpfc[clipy1:-clipy2, clip2 + medic_dlpfc.shape[1] // 2 : -clip1]
    medic_dlpfc = np.concatenate([medic_dlpfc_left, medic_dlpfc_right], axis=1)
    topup_dlpfc = np.array(Image.open(topup_dlpfc_path))
    topup_dlpfc_left = topup_dlpfc[clipy1:-clipy2, clip1 : topup_dlpfc.shape[1] // 2 - clip2]
    topup_dlpfc_right = topup_dlpfc[clipy1:-clipy2, clip2 + topup_dlpfc.shape[1] // 2 : -clip1]
    topup_dlpfc = np.concatenate([topup_dlpfc_left, topup_dlpfc_right], axis=1)
    truth_dlpfc = np.array(Image.open(truth_dlpfc_path))
    truth_dlpfc_left = truth_dlpfc[clipy1:-clipy2, clip1 : truth_dlpfc.shape[1] // 2 - clip2]
    truth_dlpfc_right = truth_dlpfc[clipy1:-clipy2, clip2 + truth_dlpfc.shape[1] // 2 : -clip1]
    truth_dlpfc = np.concatenate([truth_dlpfc_left, truth_dlpfc_right], axis=1)
    medic_occipital = np.array(Image.open(medic_occipital_path))
    medic_occipital_left = medic_occipital[clipy1:-clipy2, clip1 : medic_occipital.shape[1] // 2 - clip2]
    medic_occipital_right = medic_occipital[clipy1:-clipy2, clip2 + medic_occipital.shape[1] // 2 : -clip1]
    medic_occipital = np.concatenate([medic_occipital_left, medic_occipital_right], axis=1)
    topup_occipital = np.array(Image.open(topup_occipital_path))
    topup_occipital_left = topup_occipital[clipy1:-clipy2, clip1 : topup_occipital.shape[1] // 2 - clip2]
    topup_occipital_right = topup_occipital[clipy1:-clipy2, clip2 + topup_occipital.shape[1] // 2 : -clip1]
    topup_occipital = np.concatenate([topup_occipital_left, topup_occipital_right], axis=1)
    truth_occipital = np.array(Image.open(truth_occipital_path))
    truth_occipital_left = truth_occipital[clipy1:-clipy2, clip1 : truth_occipital.shape[1] // 2 - clip2]
    truth_occipital_right = truth_occipital[clipy1:-clipy2, clip2 + truth_occipital.shape[1] // 2 : -clip1]
    truth_occipital = np.concatenate([truth_occipital_left, truth_occipital_right], axis=1)
    medic_scan = np.array(Image.open(medic_scan_path))
    medic_scan_left = medic_scan[clipy1:-clipy2, clip1 : medic_scan.shape[1] // 2 - clip2]
    medic_scan_right = medic_scan[clipy1:-clipy2, clip2 + medic_scan.shape[1] // 2 : -clip1]
    medic_scan = np.concatenate([medic_scan_left, medic_scan_right], axis=1)
    topup_scan = np.array(Image.open(topup_scan_path))
    topup_scan_left = topup_scan[clipy1:-clipy2, clip1 : topup_scan.shape[1] // 2 - clip2]
    topup_scan_right = topup_scan[clipy1:-clipy2, clip2 + topup_scan.shape[1] // 2 : -clip1]
    topup_scan = np.concatenate([topup_scan_left, topup_scan_right], axis=1)
    truth_scan = np.array(Image.open(truth_scan_path))
    truth_scan_left = truth_scan[clipy1:-clipy2, clip1 : truth_scan.shape[1] // 2 - clip2]
    truth_scan_right = truth_scan[clipy1:-clipy2, clip2 + truth_scan.shape[1] // 2 : -clip1]
    truth_scan = np.concatenate([truth_scan_left, truth_scan_right], axis=1)

    # create a figure
    f = plt.figure(figsize=(180 * MM_TO_INCHES, 100 * MM_TO_INCHES), layout="constrained")

    # create gridspec
    gs = GridSpec(
        3, 4, left=0.005, right=0.995, bottom=0.005, top=0.975, wspace=0.15, hspace=0.01, width_ratios=[9, 9, 1, 9]
    )
    gs_cbar = GridSpecFromSubplotSpec(
        3, 3, wspace=0, hspace=0, width_ratios=[3, 2, 7], height_ratios=[1, 20, 1], subplot_spec=gs[:, 2]
    )

    # plot images
    mpl.rcParams["axes.edgecolor"] = "white"
    ax_medic_dlpfc = f.add_subplot(gs[0, 0])
    ax_medic_dlpfc.imshow(medic_dlpfc)
    draw_seed(ax_medic_dlpfc, x=0.13, y=0.62)
    ax_medic_dlpfc.set_xticks([])
    ax_medic_dlpfc.set_yticks([])
    ax_medic_dlpfc.set_title("MEDIC: Dynamic distortion correction", pad=6, loc="center")
    medic_title_pos = ax_medic_dlpfc.transAxes.inverted().transform(ax_medic_dlpfc.title.get_window_extent())
    ax_medic_dlpfc.text(
        0.5, 1, "Exemplar participant", ha="center", va="center", fontsize=5, transform=ax_medic_dlpfc.transAxes
    )
    ax_medic_dlpfc.set_xlabel("Correlation to standard: r = 0.41", labelpad=2)
    ax_topup_dlpfc = f.add_subplot(gs[0, 1])
    ax_topup_dlpfc.imshow(topup_dlpfc)
    draw_seed(ax_topup_dlpfc, x=0.13, y=0.62)
    ax_topup_dlpfc.set_xticks([])
    ax_topup_dlpfc.set_yticks([])
    ax_topup_dlpfc.set_title("TOPUP: Static distortion correction", pad=6, loc="center")
    topup_title_pos = ax_topup_dlpfc.transAxes.inverted().transform(ax_topup_dlpfc.title.get_window_extent())
    ax_topup_dlpfc.text(
        0.5, 1, "Exemplar participant", ha="center", va="center", fontsize=5, transform=ax_topup_dlpfc.transAxes
    )
    ax_topup_dlpfc.set_xlabel("Correlation to standard: r = 0.18", labelpad=2)
    ax_truth_dlpfc = f.add_subplot(gs[0, 3])
    ax_truth_dlpfc.imshow(truth_dlpfc)
    draw_seed(ax_truth_dlpfc, x=0.13, y=0.62)
    ax_truth_dlpfc.set_xticks([])
    ax_truth_dlpfc.set_yticks([])
    ax_truth_dlpfc.set_title("Standard: Low motion (TOPUP: static)", pad=6, loc="center")
    truth_title_pos = ax_truth_dlpfc.transAxes.inverted().transform(ax_truth_dlpfc.title.get_window_extent())
    ax_truth_dlpfc.text(
        0.5, 1, "Exemplar participant", ha="center", va="center", fontsize=5, transform=ax_truth_dlpfc.transAxes
    )
    ax_pos = ax_medic_dlpfc.get_position()
    f.text(
        ax_pos.x0,
        ax_pos.y1 + 0.06,
        r"$\bf{a}$    Functional connectivity (FC) seed maps: Dorsolateral prefrontal cortex (DLPFC)",
        ha="left",
        va="center",
    )

    ax_medic_occiptal = f.add_subplot(gs[1, 0])
    ax_medic_occiptal.imshow(medic_occipital)
    draw_seed(ax_medic_occiptal, x=0.44, y=0.32)
    ax_medic_occiptal.set_xticks([])
    ax_medic_occiptal.set_yticks([])
    ax_medic_occiptal.set_xlabel("Correlation to standard: r = 0.53", labelpad=2)
    f.text(
        medic_title_pos[0, 0],
        medic_title_pos[0, 1],
        "MEDIC",
        ha="left",
        va="bottom",
        fontsize=6,
        transform=ax_medic_occiptal.transAxes,
    )
    ax_topup_occipital = f.add_subplot(gs[1, 1])
    ax_topup_occipital.imshow(topup_occipital)
    draw_seed(ax_topup_occipital, x=0.44, y=0.32)
    ax_topup_occipital.set_xticks([])
    ax_topup_occipital.set_yticks([])
    ax_topup_occipital.set_xlabel("Correlation to standard: r = 0.38", labelpad=2)
    f.text(
        topup_title_pos[0, 0],
        topup_title_pos[0, 1],
        "TOPUP",
        ha="left",
        va="bottom",
        fontsize=6,
        transform=ax_topup_occipital.transAxes,
    )
    ax_truth_occipital = f.add_subplot(gs[1, 3])
    ax_truth_occipital.imshow(truth_occipital)
    draw_seed(ax_truth_occipital, x=0.44, y=0.32)
    ax_truth_occipital.set_xticks([])
    ax_truth_occipital.set_yticks([])
    f.text(
        truth_title_pos[0, 0],
        truth_title_pos[0, 1],
        "Standard",
        ha="left",
        va="bottom",
        fontsize=6,
        transform=ax_truth_occipital.transAxes,
    )

    ax_pos = ax_medic_occiptal.get_position()
    f.text(
        ax_pos.x0,
        ax_pos.y1 + 0.06,
        r"$\bf{b}$    Functional connectivity (FC) seed maps: Occipital cortex (extrastriate visual)",
        ha="left",
        va="center",
    )

    ax_medic_scan = f.add_subplot(gs[2, 0])
    ax_medic_scan.imshow(medic_scan)
    draw_seed(ax_medic_scan, x=0.20, y=0.37)
    ax_medic_scan.set_xticks([])
    ax_medic_scan.set_yticks([])
    ax_medic_scan.set_xlabel("Correlation to standard: r = 0.23", labelpad=2)
    f.text(
        medic_title_pos[0, 0],
        medic_title_pos[0, 1],
        "MEDIC",
        ha="left",
        va="bottom",
        fontsize=6,
        transform=ax_medic_scan.transAxes,
    )
    ax_topup_scan = f.add_subplot(gs[2, 1])
    ax_topup_scan.imshow(topup_scan)
    draw_seed(ax_topup_scan, x=0.20, y=0.37)
    ax_topup_scan.set_xticks([])
    ax_topup_scan.set_yticks([])
    ax_topup_scan.set_xlabel("Correlation to standard: r = 0.18", labelpad=2)
    f.text(
        topup_title_pos[0, 0],
        topup_title_pos[0, 1],
        "TOPUP",
        ha="left",
        va="bottom",
        fontsize=6,
        transform=ax_topup_scan.transAxes,
    )
    ax_truth_scan = f.add_subplot(gs[2, 3])
    ax_truth_scan.imshow(truth_scan)
    draw_seed(ax_truth_scan, x=0.20, y=0.37)
    ax_truth_scan.set_xticks([])
    ax_truth_scan.set_yticks([])
    f.text(
        truth_title_pos[0, 0],
        truth_title_pos[0, 1],
        "Standard",
        ha="left",
        va="bottom",
        fontsize=6,
        transform=ax_truth_scan.transAxes,
    )
    ax_pos = ax_medic_scan.get_position()
    f.text(
        ax_pos.x0,
        ax_pos.y1 + 0.06,
        r"$\bf{c}$    Functional connectivity (FC) seed maps: Somato-cognitive action network (SCAN)",
        ha="left",
        va="center",
    )
    mpl.rcParams["axes.edgecolor"] = "black"

    # create colorbar
    cbar_ax = f.add_subplot(gs_cbar[1, 1])
    pl = cbar_ax.imshow(
        np.array([[-0.6, 0.6], [0.6, -0.6]]), vmin=-0.6, vmax=0.6, aspect="auto", cmap=nilearn_cmaps["roy_big_bl"]
    )
    cbar = f.colorbar(
        mappable=pl,
        cax=cbar_ax,
        location="left",
        orientation="vertical",
        ticks=[-0.6, -0.3, 0, 0.3, 0.6],
    )
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.invert_yaxis()
    # create axis for colorbar
    cbar.ax.set_ylabel("Functional Connectivity z(r)", labelpad=2)

    # # for computing correlations
    # # load dconn data for medic and topup
    # low_dconn = nib.load(
    #     Path("/net/10.20.145.34/DOSENBACH02/GMT2/Andrew/HEADPOSITIONCAT/Pilot_ME_res/cifti_correlation_concat")
    #     / "MSCHD02_10run_concat_ME_MNI152_T1_2mm_Swgt_norm_bpss_resid_LR_surf_subcort_32k_fsLR_brainstem_smooth1.7_corr.dconn.nii"  # noqa
    # )
    # medic_dconn = nib.load(
    #     dataset
    #     / "derivatives"
    #     / "me_pipeline"
    #     / "sub-MSCHD02"
    #     / "ses-01wNEWPROC"
    #     / "cifti_correlation"
    #     / "sub-MSCHD02_b1_MNI152_T1_2mm_Swgt_norm_bpss_resid_LR_surf_subcort_32k_fsLR_brainstem_surfsmooth1.7_subcortsmooth1.7.dconn.nii"  # noqa
    # )
    # topup_dconn = nib.load(
    #     dataset
    #     / "derivatives"
    #     / "me_pipeline"
    #     / "sub-MSCHD02"
    #     / "ses-01wTOPUP"
    #     / "cifti_correlation"
    #     / "sub-MSCHD02_b1_MNI152_T1_2mm_Swgt_norm_bpss_resid_LR_surf_subcort_32k_fsLR_brainstem_surfsmooth1.7_subcortsmooth1.7.dconn.nii"  # noqa
    # )
    # # get the lower triangle of the dconn data
    # print("Loading dconn data...")
    # low_dconn_data = low_dconn.dataobj[:59412, :59412][np.tril_indices(59412)]
    # medic_dconn_data = medic_dconn.dataobj[:59412, :59412][np.tril_indices(59412)]
    # topup_dconn_data = topup_dconn.dataobj[:59412, :59412][np.tril_indices(59412)]
    # # remove nans
    # low_dconn_data[np.isnan(low_dconn_data)] = 0
    # medic_dconn_data[np.isnan(medic_dconn_data)] = 0
    # topup_dconn_data[np.isnan(topup_dconn_data)] = 0
    # print("Done.")
    # # compute correlations
    # print("Computing correlations...")
    # medic_corr = pearsonr(low_dconn_data, medic_dconn_data)
    # print(medic_corr)
    # topup_corr = pearsonr(low_dconn_data, topup_dconn_data)
    # print(topup_corr)
    # print("Done.")

    f.savefig(FIGURES_DIR / "head_position_concat.png", dpi=300)
    current_dir = os.getcwd()
    os.chdir(FIGURES_DIR)
    Path("figure2.png").unlink(missing_ok=True)
    Path("figure2.png").symlink_to("head_position_concat.png")
    os.chdir(current_dir)
    sns.set_theme(**GLOBAL_SETTINGS)


# figure 3
def group_template_comparison(data):
    aa_dir = Path(data)
    with open(aa_dir / "paircorr.json", "r") as f:
        data = json.load(f)

    # convert to dataframe
    df = pd.DataFrame(data)
    medic_similarities = df.MEDIC.to_numpy()
    topup_similarities = df.TOPUP.to_numpy()

    # get where medic is better and topup is better
    medic_better = medic_similarities > topup_similarities
    topup_better = medic_similarities < topup_similarities

    # get similarities where medic is better and topup is better
    medic_better_similarities_medic = medic_similarities[medic_better]
    topup_better_similarities_medic = topup_similarities[medic_better]
    medic_better_similarities_topup = medic_similarities[topup_better]
    topup_better_similarities_topup = topup_similarities[topup_better]

    # make figure
    f = plt.figure(figsize=(180 * MM_TO_INCHES, 128 * MM_TO_INCHES), layout="constrained")

    # create grid specs
    gs = GridSpec(
        1, 5, left=0.005, right=0.995, bottom=0.505, top=0.995, wspace=0.075, hspace=0, width_ratios=[9, 1, 9, 1, 9]
    )
    gs_cbar = GridSpecFromSubplotSpec(
        3, 3, wspace=0, hspace=0, width_ratios=[2, 1, 6], height_ratios=[2, 5, 2], subplot_spec=gs[3]
    )
    gs_bot = GridSpec(1, 2, left=0.04, right=0.96, bottom=0.03, top=0.47, wspace=0.25, hspace=0.1, width_ratios=[1, 2])
    gs_tstat = GridSpecFromSubplotSpec(1, 2, wspace=0.075, hspace=0, width_ratios=[9, 1], subplot_spec=gs_bot[1])
    gs_tstat_cbar = GridSpecFromSubplotSpec(
        3, 3, wspace=0, hspace=0, width_ratios=[3, 1, 9], height_ratios=[1, 10, 1], subplot_spec=gs_tstat[1]
    )

    # plot surfaces
    clip_1 = 145
    clip_2 = 145
    medic_occipital_path = DATA_DIR / "medic_occipital_20008.png"
    medic_occipital = np.array(Image.open(medic_occipital_path))
    medic_occipital_left = medic_occipital[:, clip_1 : medic_occipital.shape[1] // 2 - clip_2]
    medic_occipital_right = medic_occipital[:, clip_2 + medic_occipital.shape[1] // 2 : -clip_1]
    medic_occipital = np.concatenate((medic_occipital_left, medic_occipital_right), axis=1)
    topup_occipital_path = DATA_DIR / "topup_occipital_20008.png"
    topup_occipital = np.array(Image.open(topup_occipital_path))
    topup_occipital_left = topup_occipital[:, clip_1 : topup_occipital.shape[1] // 2 - clip_2]
    topup_occipital_right = topup_occipital[:, clip_2 + topup_occipital.shape[1] // 2 : -clip_1]
    topup_occipital = np.concatenate((topup_occipital_left, topup_occipital_right), axis=1)
    group_template_abcd_path = DATA_DIR / "group_abcd_template_surface.png"
    group_template_abcd = np.array(Image.open(group_template_abcd_path))
    group_template_abcd_left = group_template_abcd[:, clip_1 : group_template_abcd.shape[1] // 2 - clip_2]
    group_template_abcd_right = group_template_abcd[:, clip_2 + group_template_abcd.shape[1] // 2 : -clip_1]
    group_template_abcd = np.concatenate((group_template_abcd_left, group_template_abcd_right), axis=1)
    mpl.rcParams["axes.edgecolor"] = "white"
    mpl.rcParams["font.size"] = 6
    axl_medic = f.add_subplot(gs[0])
    axl_topup = f.add_subplot(gs[2])
    axl_group = f.add_subplot(gs[4])
    axl_medic.imshow(medic_occipital)
    draw_seed(axl_medic, x=0.48, y=0.7)
    axl_medic.set_xticks([])
    axl_medic.set_yticks([])
    axl_medic.set_xlabel("Correlation to standard: r = 0.44", labelpad=6)
    axl_medic.set_title("MEDIC: Dynamic distortion correction", pad=6)
    axl_medic.text(0.5, 0.5, "Participant 1", ha="center", va="center", transform=axl_medic.transAxes)
    axl_topup.imshow(topup_occipital)
    draw_seed(axl_topup, x=0.48, y=0.7)
    axl_topup.set_xticks([])
    axl_topup.set_yticks([])
    axl_topup.set_xlabel("Correlation to standard: r = 0.04", labelpad=6)
    axl_topup.set_title("TOPUP: Static distortion correction", pad=6)
    axl_topup.text(0.5, 0.5, "Participant 1", ha="center", va="center", transform=axl_topup.transAxes)
    axl_group.imshow(group_template_abcd)
    draw_seed(axl_group, x=0.48, y=0.7)
    axl_group.set_xticks([])
    axl_group.set_yticks([])
    axl_group.set_title("Group-averaged standard (TOPUP: static)", pad=6)
    axl_group.text(0.5, 0.5, "ABCD (N = 3,928)", ha="center", va="center", transform=axl_group.transAxes)
    axl_pos = axl_medic.get_position()
    f.text(
        0.01,
        axl_pos.y1 + 0.075,
        r"$\bf{a}$    Functional Connectivity (FC) seed maps: Occipital Cortex",
        ha="left",
        va="center",
        fontsize=7,
    )

    mpl.rcParams["axes.edgecolor"] = "black"
    mpl.rcParams["xtick.labelsize"] = 5
    mpl.rcParams["ytick.labelsize"] = 5
    cbar_ax = f.add_subplot(gs_cbar[1, 1])
    pl = cbar_ax.imshow(
        np.array([[-0.5, 0.5], [0.5, -0.5]]), vmin=-0.5, vmax=0.5, aspect="auto", cmap=nilearn_cmaps["roy_big_bl"]
    )
    cbar = f.colorbar(
        mappable=pl,
        cax=cbar_ax,
        location="left",
        orientation="vertical",
        ticks=[-0.5, -0.25, 0, 0.25, 0.5],
    )
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.invert_yaxis()
    # create axis for colorbar
    cbar.ax.set_ylabel("Functional Connectivity z(r)", labelpad=2)

    # plot group similarities
    settings = GLOBAL_SETTINGS.copy()
    settings["style"] = "darkgrid"
    sns.set_theme(**settings)
    mpl.rcParams["xtick.major.size"] = 0.5
    mpl.rcParams["xtick.major.pad"] = 2
    mpl.rcParams["ytick.major.size"] = 0.5
    mpl.rcParams["font.size"] = 6
    ax1 = f.add_subplot(gs_bot[0])
    clr_pal = sns.color_palette("pastel")
    sns.scatterplot(topup_better_similarities_topup, medic_better_similarities_topup, s=14, ax=ax1)
    sns.scatterplot(topup_better_similarities_medic, medic_better_similarities_medic, s=14, ax=ax1)
    sns.scatterplot([0.51], [0.33], s=14, linewidth=0.5, ax=ax1, color=clr_pal[0])
    sns.scatterplot([0.32], [0.52], s=14, linewidth=0.5, ax=ax1, color=clr_pal[1])
    ax1.text(0.515, 0.33, "Scan", ha="left", va="center", fontsize=5, transform=ax1.transData)
    ax1.text(0.325, 0.52, "Scan", ha="left", va="center", fontsize=5, transform=ax1.transData)
    ax1.axline((0, 0), slope=1, color="black", linestyle="--", linewidth=1)
    ax1.set_xlabel("Correlation (r)", labelpad=-6)
    ax1.set_ylabel("Correlation (r)", labelpad=-10)
    ax1.set_aspect("equal")
    vmax = 0.55
    vmin = 0.3
    ax1.set_xlim([vmin, vmax])
    ax1.set_ylim([vmin, vmax])
    ax1.set_xticks(np.arange(vmin, vmax, 0.05))
    ax1.set_yticks(np.arange(vmin, vmax, 0.05))
    ax1.set_xticklabels(
        [f"{x:.2f}" if np.isclose(x, vmin) or np.isclose(x, vmax) else "" for x in np.arange(vmin, vmax, 0.05)]
    )
    ax1.set_yticklabels([f"{x:.2f}" if np.isclose(x, vmax) else "" for x in np.arange(vmin, vmax, 0.05)])
    ax1.text(0.075, 0.95, "MEDIC more similar to Group Average", ha="left", va="center", transform=ax1.transAxes)
    ax1.text(0.925, 0.05, "TOPUP more similar to Group Average", ha="right", va="center", transform=ax1.transAxes)
    x = np.array([-0.1, 0.7])
    y = x
    y2 = np.ones(x.shape) * 0.6
    y3 = np.zeros(x.shape)
    colors = sns.color_palette("pastel")
    medic_color = colors[1]
    topup_color = colors[0]
    ax1.fill_between(x, y, y2, color=medic_color, alpha=0.2)
    ax1.fill_between(x, y3, y, color=topup_color, alpha=0.2)
    ax1_pos = ax1.get_position()
    f.text(
        0.01,
        ax1_pos.y1 + 0.075,
        r"$\bf{b}$    Whole-brain FC similarity to group-averaged standard",
        ha="left",
        va="center",
        fontsize=7,
    )
    sns.set_theme(**GLOBAL_SETTINGS)

    # plot t-statistic surface
    tstat_surface_path = DATA_DIR / "group_tstat_surface.png"
    tstat_surface = np.array(Image.open(tstat_surface_path))
    tstat_surface_left = tstat_surface[:, clip_1 : tstat_surface.shape[1] // 2 - clip_2]
    tstat_surface_right = tstat_surface[:, clip_2 + tstat_surface.shape[1] // 2 : -clip_1]
    tstat_surface = np.concatenate((tstat_surface_left, tstat_surface_right), axis=1)
    mpl.rcParams["axes.edgecolor"] = "white"
    mpl.rcParams["xtick.labelsize"] = 5
    mpl.rcParams["ytick.labelsize"] = 5
    ax2 = f.add_subplot(gs_tstat[0])
    ax2.imshow(tstat_surface)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.text(0.5, 0.5, "N = 185 Scans", ha="center", va="center", transform=ax2.transAxes, fontsize=6)
    mpl.rcParams["axes.edgecolor"] = "black"
    cbar_ax = f.add_subplot(gs_tstat_cbar[1, 1])
    spectral_map = plt.cm.get_cmap("Spectral")
    spectral_rmap = spectral_map.reversed()
    pl = cbar_ax.imshow(np.array([[-6, 6], [6, -6]]), vmin=-6, vmax=6, aspect="auto", cmap=spectral_rmap)
    cbar = f.colorbar(
        mappable=pl,
        cax=cbar_ax,
        location="left",
        orientation="vertical",
        ticks=[-6, -3, 0, 3, 6],
    )
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.invert_yaxis()
    # create axis for colorbar
    cbar.ax.set_ylabel("t-statistic", labelpad=2)
    ax2_pos = ax2.get_position()
    f.text(
        ax2_pos.x0,
        ax1_pos.y1 + 0.075,
        r"$\bf{c}$    Whole-brain FC similarity to group-averaged standard",
        ha="left",
        va="center",
    )
    cbar.ax.text(
        0.5,
        1.075,
        "MEDIC > TOPUP",
        ha="center",
        va="center",
        transform=cbar.ax.transAxes,
        fontsize=6,
    )
    cbar.ax.text(
        0.5,
        -0.075,
        "TOPUP > MEDIC",
        ha="center",
        va="center",
        transform=cbar.ax.transAxes,
        fontsize=6,
    )

    # # for computing correlations
    # # load dconn data for medic and topup
    # group_avg = nib.load("/data/nil-bluearc/GMT/Scott/ABCD/ABCD_4.5k_all.dconn.nii")
    # medic_dconn = nib.load(
    #     AA_DATA_DIR
    #     / "sub-20008"
    #     / "ses-51692"
    #     / "cifti_correlation"
    #     / "sub-20008_b1_MNI152_T1_2mm_Swgt_norm_bpss_resid_LR_surf_subcort_32k_fsLR_brainstem_surfsmooth1.7_subcortsmooth1.7.dconn.nii"  # noqa
    # )
    # topup_dconn = nib.load(
    #     AA_DATA_DIR
    #     / "sub-20008"
    #     / "ses-51692wTOPUP"
    #     / "cifti_correlation"
    #     / "sub-20008_b1_MNI152_T1_2mm_Swgt_norm_bpss_resid_LR_surf_subcort_32k_fsLR_brainstem_surfsmooth1.7_subcortsmooth1.7.dconn.nii"  # noqa
    # )
    # # get the lower triangle of the dconn data
    # print("Loading dconn data...")
    # group_avg_data = group_avg.dataobj[21891, :59412]
    # medic_dconn_data = medic_dconn.dataobj[21891, :59412]
    # topup_dconn_data = topup_dconn.dataobj[21891, :59412]
    # # remove nans
    # group_avg_data[np.isnan(group_avg_data)] = 0
    # medic_dconn_data[np.isnan(medic_dconn_data)] = 0
    # topup_dconn_data[np.isnan(topup_dconn_data)] = 0
    # print("Done.")
    # # compute correlations
    # print("Computing correlations...")
    # medic_corr = pearsonr(group_avg_data, medic_dconn_data)
    # print(medic_corr)
    # topup_corr = pearsonr(group_avg_data, topup_dconn_data)
    # print(topup_corr)
    # print("Done.")

    f.savefig(FIGURES_DIR / "group_template_compare.png", dpi=300)
    current_dir = os.getcwd()
    os.chdir(FIGURES_DIR)
    Path("figure3.png").unlink(missing_ok=True)
    Path("figure3.png").symlink_to("group_template_compare.png")
    os.chdir(current_dir)
    sns.set_theme(**GLOBAL_SETTINGS)


# figure 4
def fmap_comparison(data_dir):
    mpl.rcParams["axes.titlesize"] = 7
    mpl.rcParams["axes.labelsize"] = 7
    # get data
    data_path = Path(data_dir)

    # load images
    minn_example_medic_path = data_path / "UMinn_medic.png"
    minn_example_medic = equalize_hist(np.array(Image.open(minn_example_medic_path)))
    minn_example_topup_path = data_path / "UMinn_topup.png"
    minn_example_topup = equalize_hist(np.array(Image.open(minn_example_topup_path)))
    minn_example_fmap_path = data_path / "UMinn_fmap.png"
    minn_example_fmap = np.array(Image.open(minn_example_fmap_path))
    penn_example_medic_path = data_path / "Penn_medic.png"
    penn_example_medic = equalize_hist(np.array(Image.open(penn_example_medic_path)))
    penn_example_topup_path = data_path / "Penn_topup.png"
    penn_example_topup = equalize_hist(np.array(Image.open(penn_example_topup_path)))
    penn_example_fmap_path = data_path / "Penn_fmap.png"
    penn_example_fmap = np.array(Image.open(penn_example_fmap_path))
    washu_example_medic_path = data_path / "WashU_medic.png"
    washu_example_medic = equalize_hist(np.array(Image.open(washu_example_medic_path)))
    washu_example_topup_path = data_path / "WashU_topup.png"
    washu_example_topup = equalize_hist(np.array(Image.open(washu_example_topup_path)))
    washu_example_fmap_path = data_path / "WashU_fmap.png"
    washu_example_fmap = np.array(Image.open(washu_example_fmap_path))

    f = plt.figure(figsize=(180 * MM_TO_INCHES, 140 * MM_TO_INCHES), layout="constrained")
    gs = GridSpec(
        4,
        3,
        left=0.03,
        right=0.97,
        bottom=0.025,
        top=0.96,
        wspace=0.15,
        hspace=0.125,
        height_ratios=[7, 1, 7, 7]
    )
    gs_cbar = GridSpecFromSubplotSpec(
        3, 3, wspace=0, hspace=0, width_ratios=[1, 50, 1], height_ratios=[11, 5, 11], subplot_spec=gs[1, :]
    )
    ax_WashU_fmap = f.add_subplot(gs[0, 0])
    ax_WashU_medic = f.add_subplot(gs[2, 0])
    ax_WashU_topup = f.add_subplot(gs[3, 0])
    ax_UMinn_fmap = f.add_subplot(gs[0, 1])
    ax_UMinn_medic = f.add_subplot(gs[2, 1])
    ax_UMinn_topup = f.add_subplot(gs[3, 1])
    ax_Penn_fmap = f.add_subplot(gs[0, 2])
    ax_Penn_medic = f.add_subplot(gs[2, 2])
    ax_Penn_topup = f.add_subplot(gs[3, 2])

    cbar_ax = f.add_subplot(gs_cbar[1, 1])
    pl = cbar_ax.imshow(np.array([[-50, 50], [50, -50]]), vmin=-50, vmax=50, aspect="auto", cmap="icefire")
    cbar = f.colorbar(
        pl,
        cax=cbar_ax,
        location="top",
        orientation="horizontal",
        ticks=[-50, -25, 0, 25, 50],
    )
    # create axis for colorbar
    cbar.ax.set_xlabel("Field map Difference (Hz)", labelpad=2)
    alt_vmin, alt_vmax = hz_limits_to_mm(-50, 50)
    cax = cbar.ax.twiny()
    cbar.ax.xaxis.set_ticks_position("top")
    cax.xaxis.set_ticks_position("bottom")
    cax.set_xlim(alt_vmin, alt_vmax)
    cbar.ax.invert_xaxis()
    cax.invert_xaxis()
    cax.xaxis.set_label_position("bottom")
    cax.set_xlabel("Displacement difference (mm)", labelpad=1)

    # plot images
    ax_WashU_fmap.imshow(washu_example_fmap)
    ax_WashU_fmap.set_title(r"$\bf{a}$    WashU Data", pad=6, loc="left")
    ax_WashU_fmap.set_xticks([])
    ax_WashU_fmap.set_yticks([])
    ax_WashU_fmap.set_ylabel("Field Map Difference (MEDIC - TOPUP)", labelpad=4)
    ax_WashU_medic.imshow(washu_example_medic)
    ax_WashU_medic.set_xticks([])
    ax_WashU_medic.set_yticks([])
    ax_WashU_medic.set_ylabel("MEDIC", labelpad=4)
    draw_arrow(ax_WashU_medic, data_to_ax(ax_WashU_medic, (1683, 1858)), data_to_ax(ax_WashU_medic, (1457, 1691)))
    draw_arrow(ax_WashU_medic, data_to_ax(ax_WashU_medic, (2045, 1620)), data_to_ax(ax_WashU_medic, (1829, 1474)))
    ax_WashU_topup.imshow(washu_example_topup)
    ax_WashU_topup.set_xticks([])
    ax_WashU_topup.set_yticks([])
    ax_WashU_topup.set_ylabel("TOPUP", labelpad=4)
    draw_arrow(ax_WashU_topup, data_to_ax(ax_WashU_topup, (1683, 1858)), data_to_ax(ax_WashU_topup, (1457, 1691)))
    draw_arrow(ax_WashU_topup, data_to_ax(ax_WashU_topup, (2045, 1620)), data_to_ax(ax_WashU_topup, (1829, 1474)))
    ax_UMinn_fmap.imshow(minn_example_fmap)
    ax_UMinn_fmap.set_title(r"$\bf{b}$    UMinn Data", pad=6, loc="left")
    ax_UMinn_fmap.set_xticks([])
    ax_UMinn_fmap.set_yticks([])
    ax_UMinn_medic.imshow(minn_example_medic)
    draw_arrow(ax_UMinn_medic, data_to_ax(ax_UMinn_medic, (1129, 622)), data_to_ax(ax_UMinn_medic, (898, 473)))
    ax_UMinn_medic.set_xticks([])
    ax_UMinn_medic.set_yticks([])
    ax_UMinn_topup.imshow(minn_example_topup)
    ax_UMinn_topup.set_xticks([])
    ax_UMinn_topup.set_yticks([])
    draw_arrow(ax_UMinn_topup, data_to_ax(ax_UMinn_topup, (1129, 622)), data_to_ax(ax_UMinn_topup, (898, 473)))
    ax_Penn_fmap.imshow(penn_example_fmap)
    ax_Penn_fmap.set_title(r"$\bf{c}$    Penn Data", pad=6, loc="left")
    ax_Penn_fmap.set_xticks([])
    ax_Penn_fmap.set_yticks([])
    ax_Penn_medic.imshow(penn_example_medic)
    ax_Penn_medic.set_xticks([])
    ax_Penn_medic.set_yticks([])
    draw_arrow(ax_Penn_medic, data_to_ax(ax_Penn_medic, (817, 611)), data_to_ax(ax_Penn_medic, (940, 414)))
    draw_arrow(ax_Penn_medic, data_to_ax(ax_Penn_medic, (1202, 390)), data_to_ax(ax_Penn_medic, (1266, 164)))
    ax_Penn_topup.imshow(penn_example_topup)
    ax_Penn_topup.set_xticks([])
    ax_Penn_topup.set_yticks([])
    draw_arrow(ax_Penn_topup, data_to_ax(ax_Penn_topup, (817, 611)), data_to_ax(ax_Penn_topup, (940, 414)))
    draw_arrow(ax_Penn_topup, data_to_ax(ax_Penn_topup, (1202, 390)), data_to_ax(ax_Penn_topup, (1266, 164)))

    # save figure
    f.savefig(FIGURES_DIR / "fieldmap_comparison.png", dpi=300)
    current_dir = os.getcwd()
    os.chdir(FIGURES_DIR)
    Path("figure4.png").unlink(missing_ok=True)
    Path("figure4.png").symlink_to("fieldmap_comparison.png")
    os.chdir(current_dir)
    sns.set_theme(**GLOBAL_SETTINGS)


# figure 5
def spotlight_comparison(data):
    mpl.rcParams["axes.titlesize"] = 7
    mpl.rcParams["axes.labelsize"] = 7
    # load t1 and t2 t stat maps
    t1_tstat = nib.load(Path(data) / "local_corr_t1_tstat.nii.gz").get_fdata().squeeze()
    t2_tstat = nib.load(Path(data) / "local_corr_t2_tstat.nii.gz").get_fdata().squeeze()
    t1_atlas_exemplar = (
        nib.load(AA_DATA_DIR / "sub-20008" / "T1" / "atlas" / "sub-20008_T1w_debias_avg_on_MNI152_T1_2mm.nii.gz")
        .get_fdata()
        .squeeze()
    )
    t2_atlas_exemplar = (
        nib.load(AA_DATA_DIR / "sub-20008" / "T1" / "atlas" / "sub-20008_T2w_debias_avg_on_MNI152_T1_2mm.nii.gz")
        .get_fdata()
        .squeeze()
    )

    # choose slices to iterate over
    slices = np.linspace(16, t1_tstat.shape[2] - 20, 9).astype(int)[::-1]

    # create figure
    f = plt.figure(figsize=(180 * MM_TO_INCHES, 100 * MM_TO_INCHES), layout="constrained")

    # create gridspec
    gs = GridSpec(3, 3, left=0, right=0.975, bottom=0.025, top=0.95, wspace=0.02, hspace=0.02)

    # create subfigures
    subfigs = f.subfigures(1, 3, width_ratios=[1, 5, 5], wspace=0.03)

    cgs = GridSpec(1, 1, left=0.52, right=0.57, bottom=0.073, top=0.89)
    cbar_ax = subfigs[0].add_subplot(cgs[:, :])

    # create axes for each subfigure
    axes_list1 = []
    for i in range(3):
        for j in range(3):
            axes_list1.append(subfigs[1].add_subplot(gs[i, j]))
    axes_list2 = []
    for i in range(3):
        for j in range(3):
            axes_list2.append(subfigs[2].add_subplot(gs[i, j]))

    # create new cmap
    new_cmap = sns.diverging_palette(210, 30, l=70, center="dark", as_cmap=True)

    # plot slices, iterate over list
    for i, s in enumerate(slices):
        ax1 = axes_list1[i]
        ax1.imshow(t1_atlas_exemplar[..., s].T, cmap="gray", origin="lower")
        a = ax1.imshow(t1_tstat[..., s].T, cmap=new_cmap, vmin=-10, vmax=10, origin="lower", alpha=0.75)
        ax1.set_xticks([])
        ax1.set_yticks([])
        if i == 0:
            source_plot = a
        ax2 = axes_list2[i]
        ax2.imshow(t2_atlas_exemplar[..., s].T, cmap="gray", origin="lower")
        ax2.imshow(t2_tstat[..., s].T, cmap=new_cmap, vmin=-10, vmax=10, origin="lower", alpha=0.75)
        ax2.set_xticks([])
        ax2.set_yticks([])
    axes_list1[0].set_title(r"$\bf{a}$    T1w R$^2$ Spotlight", pad=4, loc="left")
    axes_list2[0].set_title(r"$\bf{b}$    T2w R$^2$ Spotlight", pad=4, loc="left")
    # create axis for colorbar
    cbar = subfigs[0].colorbar(
        source_plot,
        cax=cbar_ax,
        location="left",
        orientation="vertical",
    )
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.set_ylabel("t-statistic", labelpad=4)
    cbar.ax.text(0.5, 1.05, "MEDIC > TOPUP", ha="center", va="center", fontsize=6, transform=cbar.ax.transAxes)
    cbar.ax.text(0.5, -0.05, "TOPUP > MEDIC", ha="center", va="center", fontsize=6, transform=cbar.ax.transAxes)

    f.savefig(FIGURES_DIR / "spotlight_comparison.png", dpi=300)
    current_dir = os.getcwd()
    os.chdir(FIGURES_DIR)
    Path("figure5.png").unlink(missing_ok=True)
    Path("figure5.png").symlink_to("spotlight_comparison.png")
    os.chdir(current_dir)
    sns.set_theme(**GLOBAL_SETTINGS)


# figure 6
def alignment_metrics(data):
    # plot stats
    data = pd.read_csv(data)

    # create figures
    settings = GLOBAL_SETTINGS.copy()
    settings["style"] = "darkgrid"
    sns.set_theme(**settings)
    mpl.rcParams["axes.labelsize"] = 6
    mpl.rcParams["axes.labelpad"] = 2
    mpl.rcParams["xtick.labelsize"] = 5
    mpl.rcParams["xtick.major.pad"] = 2
    mpl.rcParams["ytick.labelsize"] = 5
    mpl.rcParams["ytick.major.pad"] = 2
    fig = plt.figure(figsize=(180 * MM_TO_INCHES, 90 * MM_TO_INCHES), layout="constrained")
    subfigs = fig.subfigures(1, 2, width_ratios=[2, 1], wspace=0.05)
    subfigs2 = subfigs[0].subfigures(2, 1, hspace=0.05, height_ratios=[1, 3])
    fig_local = subfigs2[0]
    fig_local.suptitle(r"$\bf{a}$    Local Metrics", fontsize=7, ha="left", x=0.02, weight="normal")
    axes_local = fig_local.subplots(1, 2)
    fig_global = subfigs2[1]
    fig_global.suptitle(r"$\bf{b}$    Global Metrics", fontsize=7, ha="left", x=0.02, weight="normal")
    axes_global = fig_global.subplots(3, 2)
    fig_roc = subfigs[1]
    fig_roc.suptitle(r"$\bf{c}$    Segmentation Metrics", fontsize=7, ha="left", x=0.02, weight="normal")
    axes_roc = fig_roc.subplots(4, 1)

    # create list of metrics
    metrics = [
        "local_corr_mean_t1",
        "local_corr_mean_t2",
        "corr_t1",
        "corr_t2",
        "grad_corr_t1",
        "grad_corr_t2",
        "nmi_t1",
        "nmi_t2",
        "roc_gw",
        "roc_ie",
        "roc_vw",
        "roc_cb_ie",
    ]

    # create list of titles
    titles = [
        "T1w R$^2$ Spotlight",
        "T2w R$^2$ Spotlight",
        "T1w R$^2$",
        "T2w R$^2$",
        "T1w Grad. Correlation",
        "T2w Grad. Correlation",
        "T1w NMI",
        "T2w NMI",
        "Gray/White Matter AUC",
        "Brain/Exterior AUC",
        "Ventricles/White Matter AUC",
        "Cerebellum/Exterior AUC",
    ]
    formatted_titles = [
        "T1w R$^2$\nSpotlight",
        "T2w R$^2$\nSpotlight",
        "T1w R$^2$",
        "T2w R$^2$",
        "T1w Grad.\nCorrelation",
        "T2w Grad.\nCorrelation",
        "T1w NMI",
        "T2w NMI",
        "Gray/White\nMatter AUC",
        "Brain/Exterior\nAUC",
        "Ventricles/White\nMatter AUC",
        "Cerebellum/Exterior\nAUC",
    ]

    # create list of axes for plotting
    axes_list = [
        axes_local[0],
        axes_local[1],
        axes_global[0][0],
        axes_global[0][1],
        axes_global[1][0],
        axes_global[1][1],
        axes_global[2][0],
        axes_global[2][1],
        axes_roc[0],
        axes_roc[1],
        axes_roc[2],
        axes_roc[3],
    ]

    # plot box plots
    for m, t, a in zip(metrics, formatted_titles, axes_list):
        plot_box_plot(data, m, t, a)

    # print ttest results
    table_str = "| Metric | MEDIC | TOPUP | t-statistic | p-value | df |\n"
    table_str += "| ------ | ----- | ----- | ----------- | ------- | -- |\n"
    for m, t in zip(metrics, titles):
        medic_data = data[f"{m}_medic"]
        topup_data = data[f"{m}_topup"]
        res = ttest_rel(medic_data, topup_data)
        # round stats
        medic_mean = np.round(medic_data.mean(), 3)
        medic_std = np.round(medic_data.std(), 3)
        topup_mean = np.round(topup_data.mean(), 3)
        topup_std = np.round(topup_data.std(), 3)
        p_value = np.round(res.pvalue, 3)
        p_value = p_value if p_value >= 0.001 else "<0.001"
        t_stat = np.round(res.statistic, 3)
        print(f"{t}:")
        print(f"MEDIC={medic_mean} ({medic_std}); " f"TOPUP={topup_mean} ({topup_std}); " f"p={p_value}; t={t_stat}\n")
        table_str += f"| {t} | {medic_mean} ({medic_std}) | {topup_mean} ({topup_std}) |"
        table_str += f" {t_stat} | {p_value} | {data.shape[0] - 1} |\n"
    print(table_str)

    # save figure
    fig.savefig(FIGURES_DIR / "alignment_metrics.png", dpi=300)
    current_dir = os.getcwd()
    os.chdir(FIGURES_DIR)
    Path("figure6.png").unlink(missing_ok=True)
    Path("figure6.png").symlink_to("alignment_metrics.png")
    os.chdir(current_dir)
    sns.set_theme(**GLOBAL_SETTINGS)


def resp_analysis(data):
    settings = GLOBAL_SETTINGS.copy()
    settings["style"] = "darkgrid"
    sns.set_theme(**settings)
    mpl.rcParams["axes.labelsize"] = 6
    mpl.rcParams["axes.labelpad"] = 2
    mpl.rcParams["xtick.labelsize"] = 6
    mpl.rcParams["xtick.major.pad"] = 2
    mpl.rcParams["ytick.labelsize"] = 6
    mpl.rcParams["ytick.major.pad"] = 2
    mpl.rcParams["legend.fontsize"] = 6
    mpl.rcParams["legend.title_fontsize"] = 6

    # load data
    ps_csv = sorted(Path(data).glob("power_spectra_run_*.csv"))
    r_csv = sorted(Path(data).glob("resp_data_run_*.csv"))
    power_spectra = [pd.read_csv(f).set_index("Frequency (Hz)") for f in ps_csv]
    resp_data = [pd.read_csv(f).set_index("VOLUME") for f in r_csv]

    # create figure
    f = plt.figure(figsize=(180 * MM_TO_INCHES, 140 * MM_TO_INCHES), layout="constrained")
    n_rows = len(power_spectra)
    gs = [
        GridSpec(2, 2, left=0.075, right=0.975, bottom=0.667 + 0.05, top=1.000 - 0.04, wspace=0.2, hspace=0.2),
        GridSpec(2, 2, left=0.075, right=0.975, bottom=0.333 + 0.05, top=0.667 - 0.04, wspace=0.2, hspace=0.2),
        GridSpec(2, 2, left=0.075, right=0.975, bottom=0.000 + 0.05, top=0.333 - 0.04, wspace=0.2, hspace=0.2),
    ]
    pastel = sns.color_palette("pastel")
    palette = [pastel[2], pastel[4]]
    ypos = [1.000 - 0.015, 0.667 - 0.015, 0.333 - 0.015]
    for r, l in zip(range(n_rows), ["a", "b", "c"]):
        ps = power_spectra[r]
        rd = resp_data[r]
        f.text(0.05, ypos[r], r"$\bf{" + l + r"}$    " + f"Run {r + 1}", ha="left", va="center", fontsize=7)

        # plot power spectra
        ax1 = f.add_subplot(gs[r][0, 0])
        sns.lineplot(data=ps["resp_signal"], linewidth=0.6, color=pastel[3], ax=ax1)
        ax1.set_ylim(0, 60)
        ax1.set_xlabel("")
        ax1.set_ylabel("Respiratory Belt\nPower Density", labelpad=2)
        ax1.set_xticklabels([])
        ax1.tick_params(axis="x")
        ax1.tick_params(axis="y")
        ax2 = f.add_subplot(gs[r][1, 0])
        sns.lineplot(
            data=ps[["fmap_signal", "fmap_signal_filtered"]], dashes=False, linewidth=0.6, palette=palette, ax=ax2
        )
        ax2.set_ylim(0, 60)
        ax2.legend(["Unfiltered", "Filtered"], loc="upper center")
        ax2.set_ylabel("MEDIC Field Map\nPower Density", labelpad=2)
        ax2.set_xlabel("Frequency (Hz)", labelpad=2)
        ax2.tick_params(axis="x")
        ax2.tick_params(axis="y")

        # plot resp signals
        tr = 1.761
        rd.index = rd.index * tr
        corr = np.corrcoef(rd["resp_signal"], rd["fmap_signal"])[0, 1]
        ax3 = f.add_subplot(gs[r][0, 1])
        sns.lineplot(data=rd["resp_signal"], linewidth=0.6, color=pastel[3], ax=ax3)
        ax3.set_xlabel("")
        ax3.set_ylabel("Signal from\nRespiratory Belt", labelpad=2)
        ax3.set_xticklabels([])
        ax3.tick_params(axis="x")
        ax3.tick_params(axis="y")
        ax3.set_title(f"R = {np.round(corr, 3)}")
        ax3.set_ylim(-3, 3)
        ax4 = f.add_subplot(gs[r][1, 1])
        sns.lineplot(data=rd["fmap_signal"], linewidth=0.6, color=palette[1], ax=ax4)
        ax4.set_xlabel("Time (seconds)", labelpad=2)
        ax4.set_ylabel("Signal from\nMEDIC Field Map", labelpad=2)
        ax4.tick_params(axis="x")
        ax4.tick_params(axis="y")
        ax4.set_ylim(-3, 3)
    f.savefig(FIGURES_DIR / "resp_analysis.png", dpi=300)
    current_dir = os.getcwd()
    os.chdir(FIGURES_DIR)
    Path("figure7.png").unlink(missing_ok=True)
    Path("figure7.png").symlink_to("resp_analysis.png")
    os.chdir(current_dir)
    sns.set_theme(**GLOBAL_SETTINGS)


# figure 10
def tsnr_comparision(data):
    # load tsnralignment_metrics
    tsnr_table = pd.read_csv(data)
    settings = GLOBAL_SETTINGS.copy()
    settings["style"] = "darkgrid"
    sns.set_theme(**settings)
    mpl.rcParams["axes.labelsize"] = 7
    mpl.rcParams["axes.labelpad"] = 2
    mpl.rcParams["xtick.labelsize"] = 7
    mpl.rcParams["xtick.major.pad"] = 2
    mpl.rcParams["ytick.labelsize"] = 7
    mpl.rcParams["ytick.major.pad"] = 2
    f = plt.figure(figsize=(90 * MM_TO_INCHES, 45 * MM_TO_INCHES), layout="constrained")
    ax = f.add_subplot(1, 1, 1)
    mpl.rcParams["xtick.labelsize"] = 6
    mpl.rcParams["ytick.labelsize"] = 6
    plot_box_plot(tsnr_table, "mean_tsnr_masked", "tSNR", ax)
    ax.set_xlim([0, 160])
    m = "mean_tsnr_masked"
    t = "tSNR"
    medic_data = tsnr_table[f"{m}_medic"]
    topup_data = tsnr_table[f"{m}_topup"]
    res = ttest_rel(medic_data, topup_data)
    # round stats
    medic_mean = np.round(medic_data.mean(), 3)
    medic_std = np.round(medic_data.std(), 3)
    topup_mean = np.round(topup_data.mean(), 3)
    topup_std = np.round(topup_data.std(), 3)
    p_value = np.round(res.pvalue, 3)
    p_value = p_value if p_value >= 0.001 else "<0.001"
    t_stat = np.round(res.statistic, 3)
    print(f"{t}:")
    print(f"MEDIC={medic_mean} ({medic_std}); " f"TOPUP={topup_mean} ({topup_std}); " f"p={p_value}; t={t_stat}\n")
    f.savefig(FIGURES_DIR / "tsnr.png", dpi=300)
    current_dir = os.getcwd()
    os.chdir(FIGURES_DIR)
    Path("figure10.png").unlink(missing_ok=True)
    Path("figure10.png").symlink_to("tsnr.png")
    os.chdir(current_dir)
    sns.set_theme(**GLOBAL_SETTINGS)


# figure 100
def head_position_videos(data):
    settings = GLOBAL_SETTINGS.copy()
    settings["rc"].update(
        {
            "axes.facecolor": "black",
            "figure.facecolor": "black",
            "axes.labelcolor": "white",
            "axes.titlecolor": "white",
            "text.color": "white",
            "xtick.color": "white",
            "ytick.color": "white",
        }
    )
    sns.set_theme(**settings)
    # load field map files
    medic_fieldmaps = Path(data) / "fieldmaps" / "medic_aligned"

    transient_head_position_run_idx = [7, 8, 9, 10, 11, 12, 13]

    # load transient field map runs
    transient_fieldmaps = []
    for idx in transient_head_position_run_idx:
        run = idx + 1
        transient_fieldmaps.append(nib.load(medic_fieldmaps / f"run{run:02d}" / "fmap.nii.gz"))

    labels = [
        "Neutral",
        "+Z Rotation",
        "-Z Rotation",
        "+X Rotation",
        "-X Rotation",
        "+Y Rotation",
        "-Y Rotation",
        "Neutral to +Z Rotation",
        "Neutral to -Z Rotation",
        "Neutral to +X Rotation",
        "Neutral to -X Rotation",
        "Neutral to +Y Rotation",
        "Neutral to -Y Rotation",
        "Neutral to -Z Translation",
        "-Z Translation",
    ]

    # get labels
    labels = [labels[i] for i in transient_head_position_run_idx]
    # replace space with underscores
    labels = [label.replace(" ", "_") for label in labels]

    # make transients output directory
    transients_out = Path(FIGURES_DIR) / "videos" / "transients"
    transients_out.mkdir(parents=True, exist_ok=True)

    # load motion parameters
    motion_params = []
    for idx in transient_head_position_run_idx:
        run = idx + 1
        motion_params.append(
            np.loadtxt(Path(data) / "framewise_align" / "func" / f"run{run:02d}" / f"run{run:02d}.par")
        )
        motion_params[-1][:, :3] = np.rad2deg(motion_params[-1][:, :3])

    # render transient field map videos
    def set_moco_label(motion_params):
        def set_figure_labels(fig, frame_num):
            # set label on figure
            fig.text(
                0.5,
                0.7,
                f"Frame {frame_num}"
                f"\nMotion Parameters:"
                f"\nrot-x: {motion_params[frame_num, 0]:.2f} deg"
                f"\nrot-y: {motion_params[frame_num, 1]:.2f} deg"
                f"\nrot-z: {motion_params[frame_num, 2]:.2f} deg"
                f"\ntx: {motion_params[frame_num, 3]:.2f} mm"
                f"\nty: {motion_params[frame_num, 4]:.2f} mm"
                f"\ntz: {motion_params[frame_num, 5]:.2f} mm",
                ha="center",
            )

            # return figure
            return fig

        # return function
        return set_figure_labels

    gs0 = GridSpec(1, 1, left=0.05, right=0.1, bottom=0.05, top=0.95)
    gs1 = GridSpec(
        1,
        3,
        left=0.125,
        right=0.95,
        bottom=0.05,
        top=0.95,
        hspace=0.025,
        wspace=0.025,
        width_ratios=[72, 110, 110],
    )

    def fig_callback():
        fig = plt.figure(figsize=(10, 6), layout="constrained")
        axes_list = [fig.add_subplot(gs1[0, i]) for i in range(3)]
        cbar_ax = fig.add_subplot(gs0[:, :])
        cbar_ax.axis("off")
        return fig, axes_list, cbar_ax

    for fmap, moco, label in zip(transient_fieldmaps, motion_params, labels):
        print(f"Processing {label}")
        render_dynamic_figure(
            str(transients_out / f"{label}.mp4"),
            [fmap],
            colorbar=True,
            colorbar_aspect=60,
            colorbar_pad=0,
            colorbar_labelpad=-5,
            colorbar_alt_range=True,
            colorbar_alt_labelpad=0,
            fraction=0.3,
            vmin=-100,
            vmax=150,
            colormaps="icefire",
            figure_fx=set_moco_label(moco),
            fig_callback=fig_callback,
            text_color="white",
        )
    sns.set_theme(**GLOBAL_SETTINGS)


def main():
    parser = argparse.ArgumentParser(description="script for generating paper figures")
    parser.add_argument("--figures", nargs="+", type=int, help="figures to generate, if not supplied will plot all")
    parser.add_argument("--figure_1_data", default=FIGURE1_DATA, help="path to figure 1 data")
    parser.add_argument("--figure_2_data", default=FIGURE2_DATA, help="path to figure 2 data")
    parser.add_argument("--figure_3_data", default=FIGURE3_DATA, help="path to figure 3 data")
    parser.add_argument("--figure_4_data", default=FIGURE4_DATA, help="path to figure 4 data")
    parser.add_argument("--figure_5_data", default=FIGURE5_DATA, help="path to figure 5 data")
    parser.add_argument("--figure_6_data", default=FIGURE6_DATA, help="path to figure 6 data")
    parser.add_argument("--figure_7_data", default=FIGURE7_DATA, help="path to figure 7 data")
    parser.add_argument("--figure_10_data", default=FIGURE10_DATA, help="path to figure 7 data")
    parser.add_argument("--figure_100_data", default=FIGURE100_DATA, help="path to figure 10 data")

    # get arguments
    args = parser.parse_args()

    if args.figures is None or 1 in args.figures:
        head_position_fieldmap(args.figure_1_data)

    if args.figures is None or 2 in args.figures:
        head_concatenation(args.figure_2_data)

    if args.figures is None or 3 in args.figures:
        group_template_comparison(args.figure_3_data)

    if args.figures is None or 4 in args.figures:
        fmap_comparison(args.figure_4_data)

    if args.figures is None or 5 in args.figures:
        spotlight_comparison(args.figure_5_data)

    if args.figures is None or 6 in args.figures:
        alignment_metrics(args.figure_6_data)

    if args.figures is None or 7 in args.figures:
        resp_analysis(args.figure_7_data)

    if args.figures is None or 10 in args.figures:
        tsnr_comparision(args.figure_10_data)

    if args.figures is not None and 100 in args.figures:
        head_position_videos(args.figure_100_data)

    plt.show()
