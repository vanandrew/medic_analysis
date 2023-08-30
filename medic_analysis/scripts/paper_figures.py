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
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns
from medic_analysis.common import data_plotter, render_dynamic_figure, hz_limits_to_mm
from . import (
    DATA_DIR,
    FIGURES_DIR,
    MM_TO_INCHES,
)
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps


# Set global seaborn figure settings
sns.set_theme(
    font="Inter",
    font_scale=1,
    palette="pastel",
    style="white",
    rc={
        "figure.dpi": 100,
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.titlepad": 0,
        "axes.labelsize": 8,
        "axes.labelpad": 0,
        "axes.linewidth": 0.5,
        "legend.title_fontsize": 8,
        "ytick.labelsize": 7,
        "xtick.labelsize": 7,
    },
)
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
# Alignment metrics
FIGURE5_DATA = str(DATA_DIR / "alignment_metrics.csv")
# Spotlight Analysis figure
FIGURE6_DATA = str(DATA_DIR)
# tSNR figure
FIGURE7_DATA = str(DATA_DIR / "tsnr.csv")
# dynamic field map videos
FIGURE10_DATA = "/home/usr/vana/GMT2/Andrew/HEADPOSITIONSUSTEST/derivatives"
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
    sb.set_ylabel(label, labelpad=2)
    ax.tick_params(axis="x", pad=-3)
    ax.tick_params(axis="y", pad=-3)
    return sb


# figure 1
def head_position_fieldmap(data):
    # Get the data
    output_dir = Path(data) / "derivatives"

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
    f0 = plt.figure(figsize=(180 * MM_TO_INCHES, 90 * MM_TO_INCHES), layout="constrained")

    # create a grid spec for the figure
    gs0 = GridSpec(1, 1, left=0.07, right=0.125, bottom=0.025, top=0.975)
    gs1 = GridSpec(
        3, 3, left=0.15, right=0.55, bottom=0.05, top=1, hspace=0.025, wspace=0.05, width_ratios=[1, 1.5, 1.5]
    )
    gs2 = GridSpec(
        3, 3, left=0.575, right=0.975, bottom=0.05, top=1, hspace=0.025, wspace=0.05, width_ratios=[1, 1.5, 1.5]
    )

    # create subplots
    cbar_ax = f0.add_subplot(gs0[:, :])
    cbar_ax.axis("off")
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
        figure=f0,
        colorbar=True,
        colorbar_aspect=60,
        colorbar_pad=0,
        colorbar_labelpad=0,
        colorbar_alt_range=True,
        colorbar_alt_labelpad=0,
        fraction=0.3,
        vmin=vlims[0],
        vmax=vlims[1],
        cbar_ax=cbar_ax,
        axes_list=axes_list,
    )
    sbs = f0.get_axes()
    sbs[2].set_xlabel(f"(A) {labels[1]} (15.0 deg)")
    sbs[5].set_xlabel(f"(B) {labels[2]} (9.8 deg)")
    sbs[8].set_xlabel(f"(C) {labels[3]} (10.6 deg)")
    sbs[11].set_xlabel(f"(D) {labels[4]} (13.7 deg)")
    sbs[14].set_xlabel(f"(E) {labels[5]} (10.8 deg)")
    sbs[17].set_xlabel(f"(F) {labels[6]} (8.6 deg)")
    f0.savefig(FIGURES_DIR / "fieldmap_differences.png", dpi=300)
    current_dir = os.getcwd()
    os.chdir(FIGURES_DIR)
    Path("figure1.png").unlink(missing_ok=True)
    Path("figure1.png").symlink_to("fieldmap_differences.png")
    os.chdir(current_dir)


# figure 2
def head_concatenation(data):
    # get dataset
    dataset = Path(data)

    # get the two pipeline outputs
    raw_func_path = (
        dataset / "sub-MSCHD02" / "ses-01" / "func" / "sub-MSCHD02_ses-01_task-rest_run-01_echo-1_part-mag_bold.nii.gz"
    )

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
    raw_func = nib.load(raw_func_path)
    medic_scan = Image.open(medic_scan_path)
    topup_scan = Image.open(topup_scan_path)
    truth_scan = Image.open(truth_scan_path)
    medic_dlpfc = Image.open(medic_dlpfc_path)
    topup_dlpfc = Image.open(topup_dlpfc_path)
    truth_dlpfc = Image.open(truth_dlpfc_path)
    medic_occipital = Image.open(medic_occipital_path)
    topup_occipital = Image.open(topup_occipital_path)
    truth_occipital = Image.open(truth_occipital_path)

    # create a figure
    f = plt.figure(figsize=(180 * MM_TO_INCHES, 90 * MM_TO_INCHES), layout="constrained")

    # grid spec for head motion images
    gsm = GridSpec(
        7, 3, left=0.025, right=0.3, bottom=0.05, top=0.92, wspace=0.01, hspace=0.01, width_ratios=[1, 1.5, 1.5]
    )

    # plot movement data
    # get min max
    func_min = raw_func.dataobj[..., 0].min()
    func_max = raw_func.dataobj[..., 0].max()
    # create subplots from gridspec
    axes_list = []
    for i in range(7):
        for j in range(3):
            axes_list.append(f.add_subplot(gsm[i, j]))
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
        figure=f,
        axes_list=axes_list,
    )
    sbs = axes_list
    sbs[0].set_ylabel("Frame 50", fontsize=LOWER_FONT_SIZE, labelpad=-4)
    sbs[3].set_ylabel("Frame 150", fontsize=LOWER_FONT_SIZE, labelpad=-4)
    sbs[6].set_ylabel("Frame 250", fontsize=LOWER_FONT_SIZE, labelpad=-4)
    sbs[9].set_ylabel("Frame 350", fontsize=LOWER_FONT_SIZE, labelpad=-4)
    sbs[12].set_ylabel("Frame 450", fontsize=LOWER_FONT_SIZE, labelpad=-4)
    sbs[15].set_ylabel("Frame 550", fontsize=LOWER_FONT_SIZE, labelpad=-4)
    sbs[18].set_ylabel("Frame 650", fontsize=LOWER_FONT_SIZE, labelpad=-4)
    f.text(0.1375, 0.96, "(A) High Motion Data", ha="center", va="center")

    # create gridspec for surface data
    gs_low_left = 0.325
    pad = 0.04
    pad2 = 0.07
    gs_high_right = 0.99
    width = (gs_high_right - gs_low_left - pad) / 3
    gs_low_right = gs_low_left + width
    gs_high_left = gs_low_right + pad
    gs_low = GridSpec(3, 1, left=gs_low_left, right=gs_low_right, bottom=0.225, top=0.92, wspace=pad2, hspace=0.01)
    gs_high = GridSpec(3, 2, left=gs_high_left, right=gs_high_right, bottom=0.225, top=0.92, wspace=pad2, hspace=0.01)

    # plot images
    # SCAN
    mpl.rcParams["axes.edgecolor"] = "white"
    ax_truth_scan = f.add_subplot(gs_low[0, 0])
    ax_truth_scan.imshow(truth_scan)
    ax_truth_scan.set_xticks([])
    ax_truth_scan.set_yticks([])
    ax_truth_scan.set_ylabel("Motor/SCAN")
    ax_medic_scan = f.add_subplot(gs_high[0, 0])
    ax_medic_scan.imshow(medic_scan)
    ax_medic_scan.set_xticks([])
    ax_medic_scan.set_yticks([])
    ax_topup_scan = f.add_subplot(gs_high[0, 1])
    ax_topup_scan.imshow(topup_scan)
    ax_topup_scan.set_xticks([])
    ax_topup_scan.set_yticks([])
    # DLPFC
    ax_truth_dlpfc = f.add_subplot(gs_low[1, 0])
    ax_truth_dlpfc.imshow(truth_dlpfc)
    ax_truth_dlpfc.set_xticks([])
    ax_truth_dlpfc.set_yticks([])
    ax_truth_dlpfc.set_ylabel("DLPFC")
    ax_medic_dlpfc = f.add_subplot(gs_high[1, 0])
    ax_medic_dlpfc.imshow(medic_dlpfc)
    ax_medic_dlpfc.set_xticks([])
    ax_medic_dlpfc.set_yticks([])
    ax_topup_dlpfc = f.add_subplot(gs_high[1, 1])
    ax_topup_dlpfc.imshow(topup_dlpfc)
    ax_topup_dlpfc.set_xticks([])
    ax_topup_dlpfc.set_yticks([])
    # Occipital
    ax_truth_occipital = f.add_subplot(gs_low[2, 0])
    ax_truth_occipital.imshow(truth_occipital)
    ax_truth_occipital.set_xticks([])
    ax_truth_occipital.set_yticks([])
    ax_truth_occipital.set_xlabel("TOPUP\n(Low Motion)", labelpad=1)
    ax_truth_occipital.set_ylabel("Occipital")
    ax_medic_occipital = f.add_subplot(gs_high[2, 0])
    ax_medic_occipital.imshow(medic_occipital)
    ax_medic_occipital.set_xticks([])
    ax_medic_occipital.set_yticks([])
    ax_medic_occipital.set_xlabel("MEDIC\n(High Motion), R = 0.363", labelpad=1)
    ax_topup_occipital = f.add_subplot(gs_high[2, 1])
    ax_topup_occipital.imshow(topup_occipital)
    ax_topup_occipital.set_xticks([])
    ax_topup_occipital.set_yticks([])
    ax_topup_occipital.set_xlabel("TOPUP\n(High Motion), R = 0.322", labelpad=1)
    mpl.rcParams["axes.edgecolor"] = "black"
    text_position = gs_low_left + (gs_high_right - gs_low_left) / 2
    f.text(text_position, 0.96, "(B) Surface Comparison", ha="center", va="center")

    # create axis for colorbar
    cbar_pad = 0.05
    cbar_ax = f.add_subplot(
        GridSpec(1, 1, left=gs_low_left + cbar_pad, right=gs_high_right - cbar_pad, bottom=0.025, top=0.125)[0, 0]
    )
    pl = cbar_ax.imshow(np.array([[-0.6, 0.6], [0.6, -0.6]]), vmin=-0.6, vmax=0.6, cmap=nilearn_cmaps["roy_big_bl"])
    cbar_ax.set_visible(False)
    cbar = f.colorbar(
        pl,
        ax=cbar_ax,
        location="bottom",
        orientation="horizontal",
        ticks=[-0.6, -0.3, 0, 0.3, 0.6],
        aspect=60,
        fraction=1.0,
    )
    cbar.ax.set_xlabel("Correlation")

    f.savefig(FIGURES_DIR / "head_position_concat.png", dpi=300)
    current_dir = os.getcwd()
    os.chdir(FIGURES_DIR)
    Path("figure2.png").unlink(missing_ok=True)
    Path("figure2.png").symlink_to("head_position_concat.png")
    os.chdir(current_dir)


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
    f = plt.figure(figsize=(180 * MM_TO_INCHES, 180 * MM_TO_INCHES), layout="constrained")

    # create grid spec
    gs = GridSpec(20, 20, left=0.025, right=0.975, bottom=0.03, top=0.93, wspace=0.1, hspace=0.1)

    # plot surfaces
    group_template_abcd_path = DATA_DIR / "group_abcd_template_surface.png"
    group_template_abcd = Image.open(group_template_abcd_path)
    medic_occipital_path = DATA_DIR / "medic_occipital_20008.png"
    medic_occipital = Image.open(medic_occipital_path)
    topup_occipital_path = DATA_DIR / "topup_occipital_20008.png"
    topup_occipital = Image.open(topup_occipital_path)
    mpl.rcParams["axes.edgecolor"] = "white"
    axl1 = f.add_subplot(gs[:5, :9])
    axl2 = f.add_subplot(gs[7:12, :9])
    axl3 = f.add_subplot(gs[13:18, :9])
    axl1.imshow(group_template_abcd)
    axl1.set_xticks([])
    axl1.set_yticks([])
    axl1.set_xlabel("ABCD Group Average (N = 3928)", labelpad=4)
    axl2.imshow(medic_occipital)
    axl2.set_xticks([])
    axl2.set_yticks([])
    axl2.set_xlabel("MEDIC (R = 0.361)", labelpad=4)
    axl3.imshow(topup_occipital)
    axl3.set_xticks([])
    axl3.set_yticks([])
    axl3.set_xlabel("TOPUP (R = 0.342)", labelpad=4)
    f.text(0.25, 0.97, "(A) Single Subject Comparison to Group Average", ha="center", va="center")
    mpl.rcParams["axes.edgecolor"] = "black"
    cbar_ax = f.add_subplot(gs[19:, :9])
    pl = cbar_ax.imshow(np.array([[-0.5, 0.5], [0.5, -0.5]]), vmin=-0.5, vmax=0.5, cmap=nilearn_cmaps["roy_big_bl"])
    cbar_ax.set_visible(False)
    cbar = f.colorbar(
        pl,
        ax=cbar_ax,
        location="bottom",
        orientation="horizontal",
        ticks=[-0.5, -0.25, 0, 0.25, 0.5],
        aspect=40,
        fraction=1.0,
    )
    # create axis for colorbar
    cbar.ax.set_xlabel("Correlation", labelpad=2)

    # plot group similarities
    ax1 = f.add_subplot(gs[:9, 11:])
    sns.scatterplot(topup_better_similarities_topup, medic_better_similarities_topup, ax=ax1)
    sns.scatterplot(topup_better_similarities_medic, medic_better_similarities_medic, ax=ax1)
    ax1.axline((0, 0), slope=1, color="black", linestyle="--")
    ax1.set_xlabel("TOPUP Mean Similarity (Correlation)", labelpad=4)
    ax1.set_ylabel("MEDIC Mean Similarity (Correlation)", labelpad=4)
    ax1.set_aspect("equal")
    f.text(0.75, 0.97, "(B) Group Template Comparison", ha="center", va="center")
    vmax = 0.55
    vmin = 0.3
    ax1.set_xlim([vmin, vmax])
    ax1.set_ylim([vmin, vmax])
    ax1.set_xticks(np.arange(vmin, vmax, 0.05))
    ax1.set_yticks(np.arange(vmin, vmax, 0.05))
    ax1.text(0.1, 0.9, "MEDIC more similar to Group Average", transform=ax1.transAxes)
    ax1.text(0.25, 0.1, "TOPUP more similar to Group Average", transform=ax1.transAxes)
    x = np.array([-0.1, 0.7])
    y = x
    y2 = np.ones(x.shape) * 0.6
    y3 = np.zeros(x.shape)
    ax1.fill_between(x, y, y2, color="red", alpha=0.2)
    ax1.fill_between(x, y3, y, color="blue", alpha=0.2)

    # plot t-statistic surface
    tstat_surface_path = DATA_DIR / "group_tstat_surface.png"
    tstat_surface = Image.open(tstat_surface_path)
    mpl.rcParams["axes.edgecolor"] = "white"
    ax2 = f.add_subplot(gs[13:18, 11:])
    ax2.imshow(tstat_surface)
    ax2.set_xticks([])
    ax2.set_yticks([])
    f.text(0.75, 0.375, "(C) Vertex-wise Similarity", ha="center", va="center")
    mpl.rcParams["axes.edgecolor"] = "black"
    cbar_ax = f.add_subplot(gs[19:, 11:])
    spectral_map = plt.cm.get_cmap("Spectral")
    spectral_rmap = spectral_map.reversed()
    pl = cbar_ax.imshow(np.array([[-6, 6], [6, -6]]), vmin=-6, vmax=6, cmap=spectral_rmap)
    cbar_ax.set_visible(False)
    cbar = f.colorbar(
        pl,
        ax=cbar_ax,
        location="bottom",
        orientation="horizontal",
        ticks=[-6, -3, 0, 3, 6],
        aspect=40,
        fraction=1.0,
    )
    # create axis for colorbar
    cbar.ax.set_xlabel("t-statistic (MEDIC > TOPUP)", labelpad=2)

    f.savefig(FIGURES_DIR / "group_template_compare.png", dpi=300)
    current_dir = os.getcwd()
    os.chdir(FIGURES_DIR)
    Path("figure3.png").unlink(missing_ok=True)
    Path("figure3.png").symlink_to("group_template_compare.png")
    os.chdir(current_dir)


# figure 4
def fmap_comparison(data_dir):
    # get data
    data_path = Path(data_dir)

    # load images
    minn_example_medic_path = data_path / "UMinn_medic.png"
    minn_example_medic = Image.open(minn_example_medic_path)
    minn_example_topup_path = data_path / "UMinn_topup.png"
    minn_example_topup = Image.open(minn_example_topup_path)
    minn_example_fmap_path = data_path / "UMinn_fmap.png"
    minn_example_fmap = Image.open(minn_example_fmap_path)
    penn_example_medic_path = data_path / "Penn_medic.png"
    penn_example_medic = Image.open(penn_example_medic_path)
    penn_example_topup_path = data_path / "Penn_topup.png"
    penn_example_topup = Image.open(penn_example_topup_path)
    penn_example_fmap_path = data_path / "Penn_fmap.png"
    penn_example_fmap = Image.open(penn_example_fmap_path)
    washu_example_medic_path = data_path / "WashU_medic.png"
    washu_example_medic = Image.open(washu_example_medic_path)
    washu_example_topup_path = data_path / "WashU_topup.png"
    washu_example_topup = Image.open(washu_example_topup_path)
    washu_example_fmap_path = data_path / "WashU_fmap.png"
    washu_example_fmap = Image.open(washu_example_fmap_path)

    # create figure
    width_mult = 180 * MM_TO_INCHES
    base_width = 1
    base_height = 0.828
    label_width = 0.04
    label_height = 0.04
    colorbar_width = 0.075
    colorbar_gap = 0.05
    colorbar_pad = 0.025
    cbar_labelpad = -30
    width = base_width * width_mult
    height = base_height * width_mult
    full_width = width / (1 - (label_width + colorbar_gap + colorbar_width))
    full_height = height / (1 - label_height)

    f = plt.figure(figsize=(full_width, full_height), layout="constrained")
    gs = GridSpec(
        3,
        3,
        left=label_width,
        right=(1 - (colorbar_width + colorbar_gap)),
        bottom=0,
        top=1 - label_height,
        wspace=0,
        hspace=0,
    )
    ax_WashU_fmap = f.add_subplot(gs[0, 0])
    ax_WashU_medic = f.add_subplot(gs[1, 0])
    ax_WashU_topup = f.add_subplot(gs[2, 0])
    ax_UMinn_fmap = f.add_subplot(gs[0, 1])
    ax_UMinn_medic = f.add_subplot(gs[1, 1])
    ax_UMinn_topup = f.add_subplot(gs[2, 1])
    ax_Penn_fmap = f.add_subplot(gs[0, 2])
    ax_Penn_medic = f.add_subplot(gs[1, 2])
    ax_Penn_topup = f.add_subplot(gs[2, 2])
    cgs = GridSpec(
        1,
        1,
        left=1 - colorbar_width,
        right=1,
        bottom=colorbar_pad,
        top=1 - colorbar_pad - label_height,
        wspace=colorbar_gap,
        hspace=0,
    )
    cbar_ax = f.add_subplot(cgs[0, 0])
    pl = cbar_ax.imshow(np.array([[-50, 50], [50, -50]]), vmin=-50, vmax=50, cmap="icefire")
    cbar_ax.set_visible(False)
    cbar = f.colorbar(
        pl,
        ax=cbar_ax,
        location="right",
        orientation="vertical",
        ticks=[-50, -25, 0, 25, 50],
        aspect=40,
        fraction=1.0,
    )
    # create axis for colorbar
    cbar.ax.set_ylabel("Field map Difference (Hz)", rotation=90)
    alt_vmin, alt_vmax = hz_limits_to_mm(-50, 50)
    cax = cbar.ax.twinx()
    cax.yaxis.set_ticks_position("left")
    cax.set_ylim(alt_vmin, alt_vmax)
    cax.set_ylabel("Displacement difference (mm)", labelpad=cbar_labelpad, rotation=-90)
    cbar.ax.yaxis.set_ticks_position("right")

    # plot images
    ax_WashU_fmap.imshow(washu_example_fmap)
    ax_WashU_fmap.set_title("(A) WashU", pad=3)
    ax_WashU_fmap.set_xticks([])
    ax_WashU_fmap.set_yticks([])
    ax_WashU_fmap.set_ylabel("Field Map Difference", labelpad=3)
    ax_WashU_medic.imshow(washu_example_medic)
    ax_WashU_medic.set_xticks([])
    ax_WashU_medic.set_yticks([])
    ax_WashU_medic.set_ylabel("MEDIC", labelpad=3)
    ax_WashU_topup.imshow(washu_example_topup)
    ax_WashU_topup.set_xticks([])
    ax_WashU_topup.set_yticks([])
    ax_WashU_topup.set_ylabel("TOPUP", labelpad=3)
    ax_UMinn_fmap.imshow(minn_example_fmap)
    ax_UMinn_fmap.set_title("(B) UMinn", pad=3)
    ax_UMinn_fmap.set_xticks([])
    ax_UMinn_fmap.set_yticks([])
    ax_UMinn_medic.imshow(minn_example_medic)
    ax_UMinn_medic.set_xticks([])
    ax_UMinn_medic.set_yticks([])
    ax_UMinn_topup.imshow(minn_example_topup)
    ax_UMinn_topup.set_xticks([])
    ax_UMinn_topup.set_yticks([])
    ax_Penn_fmap.imshow(penn_example_fmap)
    ax_Penn_fmap.set_title("(C) Penn", pad=3)
    ax_Penn_fmap.set_xticks([])
    ax_Penn_fmap.set_yticks([])
    ax_Penn_medic.imshow(penn_example_medic)
    ax_Penn_medic.set_xticks([])
    ax_Penn_medic.set_yticks([])
    ax_Penn_topup.imshow(penn_example_topup)
    ax_Penn_topup.set_xticks([])
    ax_Penn_topup.set_yticks([])

    # save figure
    f.savefig(FIGURES_DIR / "fieldmap_comparison.png", dpi=300)
    current_dir = os.getcwd()
    os.chdir(FIGURES_DIR)
    Path("figure4.png").unlink(missing_ok=True)
    Path("figure4.png").symlink_to("fieldmap_comparison.png")
    os.chdir(current_dir)


# figure 5
def spotlight_comparison(data):
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
    f = plt.figure(figsize=(180 * MM_TO_INCHES, 90 * MM_TO_INCHES), layout="constrained")

    # create gridspec
    gs = GridSpec(3, 3, left=0.025, right=0.975, bottom=0.025, top=0.95, wspace=0.03, hspace=0.03)

    # create subfigures
    subfigs = f.subfigures(1, 3, width_ratios=[1, 5, 5], wspace=0.03)

    cgs = GridSpec(1, 1, left=0.025, right=0.975, bottom=0.025, top=0.95, wspace=0.03, hspace=0.03)
    cbar_ax = subfigs[0].add_subplot(cgs[:, :])
    cbar_ax.axis("off")

    # create axes for each subfigure
    axes_list1 = []
    for i in range(3):
        for j in range(3):
            axes_list1.append(subfigs[1].add_subplot(gs[i, j]))
    axes_list2 = []
    for i in range(3):
        for j in range(3):
            axes_list2.append(subfigs[2].add_subplot(gs[i, j]))

    # plot slices, iterate over list
    for i, s in enumerate(slices):
        ax1 = axes_list1[i]
        ax1.imshow(t1_atlas_exemplar[..., s].T, cmap="gray", origin="lower")
        a = ax1.imshow(t1_tstat[..., s].T, cmap="icefire", vmin=-10, vmax=10, origin="lower", alpha=0.75)
        ax1.set_xticks([])
        ax1.set_yticks([])
        if i == 0:
            source_plot = a
        ax2 = axes_list2[i]
        ax2.imshow(t2_atlas_exemplar[..., s].T, cmap="gray", origin="lower")
        ax2.imshow(t2_tstat[..., s].T, cmap="icefire", vmin=-10, vmax=10, origin="lower", alpha=0.75)
        ax2.set_xticks([])
        ax2.set_yticks([])
    axes_list1[1].set_title("(A) T1w $R^2$ Spotlight", pad=4)
    axes_list2[1].set_title("(B) T2w $R^2$ Spotlight", pad=4)
    # create axis for colorbar
    cbar = subfigs[0].colorbar(
        source_plot,
        ax=cbar_ax,
        location="left",
        orientation="vertical",
        aspect=40,
        pad=-0.5,
        fraction=1.0,
    )
    cbar.ax.set_ylabel("t-statistic (MEDIC > TOPUP)", labelpad=-1)

    f.savefig(FIGURES_DIR / "spotlight_comparison.png", dpi=300)
    current_dir = os.getcwd()
    os.chdir(FIGURES_DIR)
    Path("figure5.png").unlink(missing_ok=True)
    Path("figure5.png").symlink_to("spotlight_comparison.png")
    os.chdir(current_dir)


# figure 6
def alignment_metrics(data):
    # plot stats
    data = pd.read_csv(data)

    # create figures
    mpl.rcParams["axes.labelsize"] = 6
    mpl.rcParams["xtick.labelsize"] = 6
    mpl.rcParams["ytick.labelsize"] = 6
    fig = plt.figure(figsize=(180 * MM_TO_INCHES, 90 * MM_TO_INCHES), layout="constrained")
    subfigs = fig.subfigures(1, 2, width_ratios=[2, 1], wspace=0.025)
    subfigs2 = subfigs[0].subfigures(2, 1, hspace=0.05, height_ratios=[3, 1])
    fig_global = subfigs2[0]
    fig_global.suptitle("(A) Global Metrics", fontsize=8, weight="normal")
    axes_global = fig_global.subplots(3, 2)
    fig_local = subfigs2[1]
    fig_local.suptitle("(B) Local Metrics", fontsize=8, weight="normal")
    axes_local = fig_local.subplots(1, 2)
    fig_roc = subfigs[1]
    fig_roc.suptitle("(C) Segmentation Metrics", fontsize=8, weight="normal")
    axes_roc = fig_roc.subplots(4, 1)

    # plot global metrics
    plot_box_plot(data, "corr_t1", "T1w\nCorrelation$^2$", axes_global[0][0])
    plot_box_plot(data, "corr_t2", "T2w\nCorrelation$^2$", axes_global[0][1])
    plot_box_plot(data, "grad_corr_t1", "T1wGrad.\nCorrelation", axes_global[1][0])
    plot_box_plot(data, "grad_corr_t2", "T2w Grad.\nCorrelation", axes_global[1][1])
    plot_box_plot(data, "nmi_t1", "T1w NMI", axes_global[2][0])
    plot_box_plot(data, "nmi_t2", "T2w NMI", axes_global[2][1])

    # plot local metrics
    plot_box_plot(data, "local_corr_mean_t1", "T1w $R^2$\nSpotlight", axes_local[0])
    plot_box_plot(data, "local_corr_mean_t2", "T2w $R^2$\nSpotlight", axes_local[1])

    # plot ROC metrics
    plot_box_plot(data, "roc_gw", "Gray/White\nMatter AUC", axes_roc[0])
    plot_box_plot(data, "roc_ie", "Brain/Exterior\nAUC", axes_roc[1])
    plot_box_plot(data, "roc_vw", "Ventricles/White\nMatter AUC", axes_roc[2])
    plot_box_plot(data, "roc_cb_ie", "Cerebellum/Exterior\nAUC", axes_roc[3])

    # save figure
    fig.savefig(FIGURES_DIR / "alignment_metrics.png", dpi=300)
    current_dir = os.getcwd()
    os.chdir(FIGURES_DIR)
    Path("figure6.png").unlink(missing_ok=True)
    Path("figure6.png").symlink_to("alignment_metrics.png")
    os.chdir(current_dir)
    mpl.rcParams["axes.labelsize"] = 8
    mpl.rcParams["xtick.labelsize"] = 7
    mpl.rcParams["ytick.labelsize"] = 7


# figure 7
def tsnr_comparision(data):
    # load tsnr
    tsnr_table = pd.read_csv(data)
    mpl.rcParams["axes.labelsize"] = 6
    mpl.rcParams["xtick.labelsize"] = 6
    mpl.rcParams["ytick.labelsize"] = 6
    f = plt.figure(figsize=(90 * MM_TO_INCHES, 45 * MM_TO_INCHES), layout="constrained")
    ax = f.add_subplot(1, 1, 1)
    plot_box_plot(tsnr_table, "mean_tsnr_masked", "tSNR", ax)
    ax.set_xlim([0, 160])
    f.savefig(FIGURES_DIR / "tsnr.png", dpi=300)
    current_dir = os.getcwd()
    os.chdir(FIGURES_DIR)
    Path("figure7.png").unlink(missing_ok=True)
    Path("figure7.png").symlink_to("tsnr.png")
    os.chdir(current_dir)
    mpl.rcParams["axes.labelsize"] = 8
    mpl.rcParams["xtick.labelsize"] = 7
    mpl.rcParams["ytick.labelsize"] = 7


# figure 10
def head_position_videos(data):
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

    sns.set_theme(
        font="Inter",
        font_scale=1,
        palette="dark",
        style="white",
        rc={
            "axes.facecolor": "black",
            "figure.facecolor": "black",
            "axes.labelcolor": "white",
            "axes.titlecolor": "white",
            "text.color": "white",
            "xtick.color": "white",
            "ytick.color": "white",
        },
    )

    for fmap, moco, label in zip(transient_fieldmaps, motion_params, labels):
        render_dynamic_figure(
            str(transients_out / f"{label}.mp4"),
            [fmap],
            colorbar=True,
            colorbar_alt_range=True,
            colorbar_pad=0.35,
            figure_fx=set_moco_label(moco),
            text_color="white",
            figsize=(8, 9),
        )


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
    parser.add_argument("--figure_10_data", default=FIGURE10_DATA, help="path to figure 10 data")

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
        spotlight_comparison(args.figure_6_data)

    if args.figures is None or 6 in args.figures:
        alignment_metrics(args.figure_5_data)

    if args.figures is None or 7 in args.figures:
        tsnr_comparision(args.figure_7_data)

    if args.figures is not None and 10 in args.figures:
        head_position_videos(args.figure_10_data)

    plt.show()
