from memori.pathman import PathManager as PathMan
import shutil
import numpy as np
import nibabel as nib
from IPython import embed
from warpkit.utilities import corr2_coeff
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from medic_analysis.common import FIGURE_OUT, data_plotter


DATA_PATH = PathMan("/home/usr/vana/GMT/David/MEDIC")

GROUP_TEMPLATE = PathMan("/home/usr/vana/GMT2/Andrew/120_Network_templates_erode3.dtseries.nii")

UPENN_DATA = PathMan("/home/usr/vana/GMT2/Andrew/UPenn/derivatives/me_pipeline/")

ASD_ADHD_DATA = PathMan("/home/usr/vana/Daenerys/ASD_ADHD/NP1173/derivatives/me_pipeline2")

RESPDATA = PathMan("/home/usr/vana/GMT2/Andrew/RESPTEST/derivatives/me_pipeline")

ASD_ADHD_STATS = PathMan("/home/usr/vana/GMT/David/MEDIC/MEDIC_TOPUP_FullBrain.dat")


def main():
    # output directory
    output_dir = PathMan("/home/usr/vana/GMT2/Andrew/MSCHD02_GROUP_CORR_ANALYSIS")
    output_dir.mkdir(exist_ok=True)

    # stats directory
    stats_dir = PathMan("/home/usr/vana/GMT2/Andrew/MEDIC_STATS")
    stats_dir.mkdir(exist_ok=True)

    # load dataset
    dataset = ASD_ADHD_DATA
    TOPUPNAME = "wTOPUP"

    # plot stats
    data = pd.read_csv(ASD_ADHD_STATS)

    # get subset of dataframe and melt for boxplots
    t1_corr = (
        data[["MEDICT1Corr", "TOPUPT1Corr"]]
        .rename(columns={"MEDICT1Corr": "MEDIC", "TOPUPT1Corr": "TOPUP"})
        .melt(var_name="T1w Correlation")
    )
    t2_corr = (
        data[["MEDICT2Corr", "TOPUPT2Corr"]]
        .rename(columns={"MEDICT2Corr": "MEDIC", "TOPUPT2Corr": "TOPUP"})
        .melt(var_name="T2w Correlation")
    )
    t1_grad_corr = (
        data[["MEDICT1GradCorr", "TOPUPT1GradCorr"]]
        .rename(columns={"MEDICT1GradCorr": "MEDIC", "TOPUPT1GradCorr": "TOPUP"})
        .melt(var_name="T1w Grad. Correlation")
    )
    t2_grad_corr = (
        data[["MEDICT2GradCorr", "TOPUPT2GradCorr"]]
        .rename(columns={"MEDICT2GradCorr": "MEDIC", "TOPUPT2GradCorr": "TOPUP"})
        .melt(var_name="T2w Grad. Correlation")
    )
    t1_nmi = (
        data[["MEDICT1NMI", "TOPUPT1NMI"]]
        .rename(columns={"MEDICT1NMI": "MEDIC", "TOPUPT1NMI": "TOPUP"})
        .melt(var_name="T1w NMI")
    )
    t2_nmi = (
        data[["MEDICT2NMI", "TOPUPT2NMI"]]
        .rename(columns={"MEDICT2NMI": "MEDIC", "TOPUPT2NMI": "TOPUP"})
        .melt(var_name="T2w NMI")
    )
    t1_spotlight = (
        data[["MEDICT1Spotlight", "TOPUPT1Spotlight"]]
        .rename(columns={"MEDICT1Spotlight": "MEDIC", "TOPUPT1Spotlight": "TOPUP"})
        .melt(var_name="T1w Spotlight")
    )
    t2_spotlight = (
        data[["MEDICT2Spotlight", "TOPUPT2Spotlight"]]
        .rename(columns={"MEDICT2Spotlight": "MEDIC", "TOPUPT2Spotlight": "TOPUP"})
        .melt(var_name="T2w Spotlight")
    )
    roc_ie = (
        data[["MEDICROCIE", "TOPUPROCIE"]]
        .rename(columns={"MEDICROCIE": "MEDIC", "TOPUPROCIE": "TOPUP"})
        .melt(var_name="Brain/Exterior AUC")
    )
    roc_gw = (
        data[["MEDICROCGW", "TOPUPROCGW"]]
        .rename(columns={"MEDICROCGW": "MEDIC", "TOPUPROCGW": "TOPUP"})
        .melt(var_name="Gray/White Matter AUC")
    )
    roc_vw = (
        data[["MEDICROCVW", "TOPUPROCVW"]]
        .rename(columns={"MEDICROCVW": "MEDIC", "TOPUPROCVW": "TOPUP"})
        .melt(var_name="Ventricles/White Matter AUC")
    )
    tSNR = (
        data[["MEDICTSNR", "TOPUPTSNR"]]
        .rename(columns={"MEDICTSNR": "MEDIC", "TOPUPTSNR": "TOPUP"})
        .melt(var_name="tSNR")
    )

    def plot_box_plot(data, variable, ax):
        sb = sns.boxplot(data=data, x="value", y=variable, order=["TOPUP", "MEDIC"], ax=ax)
        sb.set_xlabel("")
        return sb

    # create figures
    fig = plt.figure(figsize=(16, 8), layout="constrained")
    subfigs = fig.subfigures(1, 2, wspace=0.1)
    subfigs2 = subfigs[0].subfigures(2, 1, hspace=0.1, height_ratios=[3, 1])
    fig_global = subfigs2[0]
    fig_global.suptitle("(A) Global Metrics")
    axes_global = fig_global.subplots(3, 2)
    fig_local = subfigs2[1]
    fig_local.suptitle("(B) Local Metrics")
    axes_local = fig_local.subplots(1, 2)
    fig_roc = subfigs[1]
    fig_roc.suptitle("(C) Segmentation Metrics")
    axes_roc = fig_roc.subplots(3, 1)

    # plot global metrics
    plot_box_plot(t1_corr, "T1w Correlation", axes_global[0][0])
    plot_box_plot(t2_corr, "T2w Correlation", axes_global[0][1])
    plot_box_plot(t1_grad_corr, "T1w Grad. Correlation", axes_global[1][0])
    plot_box_plot(t2_grad_corr, "T2w Grad. Correlation", axes_global[1][1])
    plot_box_plot(t1_nmi, "T1w NMI", axes_global[2][0])
    plot_box_plot(t2_nmi, "T2w NMI", axes_global[2][1])

    # plot local metrics
    plot_box_plot(t1_spotlight, "T1w Spotlight", axes_local[0])
    plot_box_plot(t2_spotlight, "T2w Spotlight", axes_local[1])

    # plot ROC metrics
    plot_box_plot(roc_ie, "Brain/Exterior AUC", axes_roc[0])
    plot_box_plot(roc_gw, "Gray/White Matter AUC", axes_roc[1])
    plot_box_plot(roc_vw, "Ventricles/White Matter AUC", axes_roc[2])

    # save figure
    fig.savefig(FIGURE_OUT / "group_comparison.png", dpi=300, bbox_inches="tight")

    # make figure for tSNR
    fig_tsnr = plt.figure(figsize=(16, 8), layout="constrained")
    fig_tsnr.set_facecolor("black")
    subfigs_tsnr = fig_tsnr.subfigures(1, 2)

    # load tSNR data for sub-20001
    # load the run 2 data
    # copy over data to local
    tSNR_MEDIC = PathMan(
        ASD_ADHD_DATA
        / "sub-20001"
        / "ses-50654"
        / "bold2"
        / "sub-20001_b2_faln_xr3d_uwrp_on_MNI152_T1_2mm_Swgt_norm_SNR.nii.gz",
    )
    tSNR_TOPUP = PathMan(
        ASD_ADHD_DATA
        / "sub-20001"
        / "ses-50654wTOPUP"
        / "bold2"
        / "sub-20001_b2_faln_xr3d_uwrp_on_MNI152_T1_2mm_Swgt_norm_SNR.nii.gz",
    )
    # tSNR_MEDIC = PathMan("sub-20001_b2_faln_xr3d_uwrp_on_MNI152_T1_2mm_Swgt_norm_SNR.nii.gz")
    tSNR_MEDIC = nib.load(tSNR_MEDIC.path).get_fdata()
    # tSNR_TOPUP = PathMan("sub-20001_b2_faln_xr3d_uwrp_on_MNI152_T1_2mm_Swgt_norm_SNR.nii.gz")
    tSNR_TOPUP = nib.load(tSNR_TOPUP.path).get_fdata()
    fig = data_plotter(
        [tSNR_TOPUP, tSNR_MEDIC],
        colormaps=["rocket", "rocket"],
        vmin=[0, 0],
        vmax=[150, 150],
        colorbar=True,
        colorbar_label="tSNR",
        colorbar_pad=0.1,
        slices=(55, 55, 32),
        figure=subfigs_tsnr[0],
        text_color="white",
    )
    fig.set_facecolor("black")
    sbs = fig.get_axes()
    sbs[1].set_title("(A) tSNR (TOPUP)", color="white", loc="center", y=-0.5)
    sbs[4].set_title("(B) tSNR (MEDIC)", color="white", loc="center", y=-0.5)

    # box plot for group tSNR
    # add subfig
    subfigs_tsnr[1].set_facecolor("black")
    subfigs_tsnrbox = subfigs_tsnr[1].subfigures(2, 1)
    fig = data_plotter(
        [tSNR_MEDIC - tSNR_TOPUP],
        colormaps=["icefire"],
        vmin=[-50],
        vmax=[50],
        colorbar2=True,
        colorbar2_label="tSNR Difference",
        colorbar2_source_idx=(0, 1),
        colorbar2_pad=0.1,
        slices=(55, 55, 32),
        figure=subfigs_tsnrbox[0],
        text_color="white",
    )
    fig.set_facecolor("black")
    sbs = fig.get_axes()
    sbs[1].set_title("(C) tSNR Difference (MEDIC - TOPUP)", color="white", loc="center", y=-0.5)
    subfigs_tsnrbox2 = subfigs_tsnrbox[1].subfigures(3, 2, height_ratios=[3, 18, 1], width_ratios=[19, 1])
    ax_tsnr = subfigs_tsnrbox2[1, 0].subplots(1, 1)
    plot_box_plot(tSNR, "tSNR", ax_tsnr)
    ax_tsnr.set_title("(D) tSNR Group Comparison", color="black", loc="center", y=-0.4)
    fig_tsnr.savefig(FIGURE_OUT / "group_tsnr.png", dpi=300, bbox_inches="tight")
    plt.show()
