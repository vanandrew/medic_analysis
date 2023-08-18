"""Main script for generating paper figures."""
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from medic_analysis.common import data_plotter
from . import (
    DATA_DIR,
    FIGURES_DIR,
    MM_TO_INCHES,
)
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps


# Set global seaborn figure settings
sns.set(
    font="Lato",
    font_scale=1,
    palette="pastel",
    style="white",
)


# Default paths for data
FIGURE1_DATA = "/home/usr/vana/GMT2/Andrew/HEADPOSITIONSUSTEST"
FIGURE2_DATA = "/home/usr/vana/GMT2/Andrew/HEADPOSITIONCAT"


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
    gs0 = GridSpec(3, 1, left=0.1, right=0.15, bottom=0.05, top=0.95)
    gs1 = GridSpec(3, 3, left=0.2, right=0.55, bottom=0.05, top=0.95, hspace=0.1, wspace=0.05)
    gs2 = GridSpec(3, 3, left=0.6, right=0.95, bottom=0.05, top=0.95, hspace=0.1, wspace=0.05)

    # create subplots
    cbar_ax = f0.add_subplot(gs0[:, 0])
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
        colorbar_aspect=30,
        colorbar_pad=0,
        colorbar_labelpad=0,
        colorbar_alt_range=True,
        colorbar_alt_labelpad=0,
        fraction=0.3,
        vmin=vlims[0],
        vmax=vlims[1],
        cbar_ax=cbar_ax,
        axes_list=axes_list,
        fontsize=8,
    )
    fontsize = 8
    sbs = f0.get_axes()
    sbs[2].set_xlabel(f"(A) {labels[1]} (15.0 deg)", fontsize=fontsize)
    sbs[5].set_xlabel(f"(B) {labels[2]} (9.8 deg)", fontsize=fontsize)
    sbs[8].set_xlabel(f"(C) {labels[3]} (10.6 deg)", fontsize=fontsize)
    sbs[11].set_xlabel(f"(D) {labels[4]} (13.7 deg)", fontsize=fontsize)
    sbs[14].set_xlabel(f"(E) {labels[5]} (10.8 deg)", fontsize=fontsize)
    sbs[17].set_xlabel(f"(F) {labels[6]} (8.6 deg)", fontsize=fontsize)
    # f0.suptitle("Motion-dependent field map differences (Position - Neutral Position)")
    f0.savefig(FIGURES_DIR / "fieldmap_differences.png", dpi=300, bbox_inches="tight")


def head_concatenation(data):
    # get dataset
    dataset = Path(data)

    # get the two pipeline outputs
    output = dataset / "derivatives" / "me_pipeline" / "sub-MSCHD02"
    raw_func_path = (
        dataset / "sub-MSCHD02" / "ses-01" / "func" / "sub-MSCHD02_ses-01_task-rest_run-01_echo-1_part-mag_bold.nii.gz"
    )
    medic_func_path = (
        output / "ses-01wNEWPROC" / "bold1" / "sub-MSCHD02_b1_faln_xr3d_uwrp_on_MNI152_T1_2mm_Swgt_norm.nii"
    )
    topup_func_path = output / "ses-01wTOPUP" / "bold1" / "sub-MSCHD02_b1_faln_xr3d_uwrp_on_MNI152_T1_2mm_Swgt_norm.nii"
    medic_fmaps_path = output / "ses-01wNEWPROC" / "bold1" / "MEDIC" / "sub-MSCHD02_b1_fieldmaps.nii"
    topup_mag_path = output / "ses-01wTOPUP" / "SEFM" / "sub-MSCHD02_sefm_Grp1_2.nii"
    topup_mag_path_2 = output / "ses-01wTOPUP" / "SEFM" / "sub-MSCHD02_sefm_Grp1_1.nii"
    topup_fmaps_path = output / "ses-01wTOPUP" / "SEFM" / "sub-MSCHD02_sefm_Grp1_FMAP.nii"

    # load medic and topup workbench screenshots
    medic_scan_path = dataset / "medic_scan.png"
    topup_scan_path = dataset / "topup_scan.png"
    medic_dlpfc_path = dataset / "medic_dlpfc.png"
    topup_dlpfc_path = dataset / "topup_dlpfc.png"
    medic_occipital_path = dataset / "medic_occipital.png"
    topup_occipital_path = dataset / "topup_occipital.png"

    # load data
    raw_func = nib.load(raw_func_path)
    # topup_mag = nib.load(topup_mag_path)
    # topup_mag_2 = nib.load(topup_mag_path_2)
    # medic_func = nib.load(medic_func_path)
    # topup_func = nib.load(topup_func_path)
    # medic_fmaps = nib.load(medic_fmaps_path)
    # topup_fmaps = nib.load(topup_fmaps_path)
    medic_scan = Image.open(medic_scan_path)
    topup_scan = Image.open(topup_scan_path)
    medic_dlpfc = Image.open(medic_dlpfc_path)
    topup_dlpfc = Image.open(topup_dlpfc_path)
    medic_occipital = Image.open(medic_occipital_path)
    topup_occipital = Image.open(topup_occipital_path)

    # create a figure
    f = plt.figure(figsize=(180 * MM_TO_INCHES, 90 * MM_TO_INCHES), layout="constrained")
    fontsize = 8

    # create subfigures
    subfigs = f.subfigures(1, 2, width_ratios=[2.25, 1], wspace=0.15)

    # create gridspecs
    gs0 = GridSpec(3, 2, left=0.22, right=1.0, bottom=0.08, top=0.95, wspace=0.01, hspace=0.01)
    gs1 = GridSpec(
        7, 3, left=0.0, right=1.0, bottom=0.08, top=0.95, wspace=0.01, hspace=0.01, width_ratios=[1, 1.5, 1.5]
    )

    # create axis for colorbar
    cbar_ax = subfigs[0].add_subplot(GridSpec(1, 1, left=0.0, right=0.15, bottom=0.1, top=0.95)[0, 0])
    pl = cbar_ax.imshow(np.array([[-0.6, 0.6], [0.6, -0.6]]), vmin=-0.6, vmax=0.6, cmap=nilearn_cmaps["roy_big_bl"])
    cbar_ax.set_visible(False)
    cbar = subfigs[0].colorbar(
        pl,
        ax=cbar_ax,
        location="left",
        orientation="vertical",
        ticks=[-0.6, -0.3, 0, 0.3, 0.6],
        pad=0,
        aspect=40,
        fraction=1.0,
    )
    cbar.ax.set_ylabel("Correlation", fontsize=fontsize, labelpad=0)
    cbar.ax.tick_params(labelsize=fontsize)

    # plot images
    # SCAN
    sns.set(style="dark")
    ax_medic_scan = subfigs[0].add_subplot(gs0[0, 0])
    ax_medic_scan.imshow(medic_scan)
    ax_medic_scan.set_xticks([])
    ax_medic_scan.set_yticks([])
    ax_medic_scan.set_ylabel("Motor/SCAN", fontsize=fontsize)
    ax_topup_scan = subfigs[0].add_subplot(gs0[0, 1])
    ax_topup_scan.imshow(topup_scan)
    ax_topup_scan.set_xticks([])
    ax_topup_scan.set_yticks([])
    # DLPFC
    ax_medic_dlpfc = subfigs[0].add_subplot(gs0[1, 0])
    ax_medic_dlpfc.imshow(medic_dlpfc)
    ax_medic_dlpfc.set_xticks([])
    ax_medic_dlpfc.set_yticks([])
    ax_medic_dlpfc.set_ylabel("DLPFC", fontsize=fontsize)
    ax_topup_dlpfc = subfigs[0].add_subplot(gs0[1, 1])
    ax_topup_dlpfc.imshow(topup_dlpfc)
    ax_topup_dlpfc.set_xticks([])
    ax_topup_dlpfc.set_yticks([])
    # Occipital
    ax_medic_occipital = subfigs[0].add_subplot(gs0[2, 0])
    ax_medic_occipital.imshow(medic_occipital)
    ax_medic_occipital.set_xticks([])
    ax_medic_occipital.set_yticks([])
    ax_medic_occipital.set_ylabel("Occipital", fontsize=fontsize)
    ax_medic_occipital.set_xlabel("MEDIC", fontsize=fontsize)
    ax_topup_occipital = subfigs[0].add_subplot(gs0[2, 1])
    ax_topup_occipital.imshow(topup_occipital)
    ax_topup_occipital.set_xticks([])
    ax_topup_occipital.set_yticks([])
    ax_topup_occipital.set_xlabel("TOPUP", fontsize=fontsize)
    sns.set(style="white")
    subfigs[0].suptitle("(A) Surface Comparison", verticalalignment="bottom", x=0.62, y=0, fontsize=fontsize)

    # plot movement data
    # get min max
    func_min = raw_func.dataobj[..., 0].min()
    func_max = raw_func.dataobj[..., 0].max()
    # create subplots from gridspec
    axes_list = []
    for i in range(7):
        for j in range(3):
            axes_list.append(subfigs[1].add_subplot(gs1[i, j]))
    # plot data
    data_plotter(
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
        figure=subfigs[1],
        axes_list=axes_list,
    )
    sbs = subfigs[1].get_axes()
    fontsize2 = 4
    labelpad = 3.0
    sbs[0].set_ylabel("Frame 50", fontsize=fontsize2, labelpad=labelpad, rotation=0)
    sbs[3].set_ylabel("Frame 150", fontsize=fontsize2, labelpad=labelpad, rotation=0)
    sbs[6].set_ylabel("Frame 250", fontsize=fontsize2, labelpad=labelpad, rotation=0)
    sbs[9].set_ylabel("Frame 350", fontsize=fontsize2, labelpad=labelpad, rotation=0)
    sbs[12].set_ylabel("Frame 450", fontsize=fontsize2, labelpad=labelpad, rotation=0)
    sbs[15].set_ylabel("Frame 550", fontsize=fontsize2, labelpad=labelpad, rotation=0)
    sbs[18].set_ylabel("Frame 650", fontsize=fontsize2, labelpad=labelpad, rotation=0)
    subfigs[1].suptitle("(B) Functional Data", verticalalignment="bottom", y=0, fontsize=fontsize)

    f.savefig(FIGURES_DIR / "head_position_concat.png", dpi=300, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="script for generating paper figures")
    parser.add_argument("--figures", nargs="+", type=int, help="figures to generate, if not supplied will plot all")
    parser.add_argument("--figure_1_data", default=FIGURE1_DATA, help="path to figure 1 data")
    parser.add_argument("--figure_2_data", default=FIGURE2_DATA, help="path to figure 2 data")

    # get arguments
    args = parser.parse_args()

    if args.figures is None or 1 in args.figures:
        head_position_fieldmap(args.figure_1_data)

    if args.figures is None or 2 in args.figures:
        head_concatenation(args.figure_2_data)

    plt.show()
