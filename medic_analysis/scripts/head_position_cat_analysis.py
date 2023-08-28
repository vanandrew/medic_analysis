from pathlib import Path
from PIL import Image
import nibabel as nib
import numpy as np
import seaborn as sns
from medic_analysis.common import data_plotter, plt, FIGURE_OUT


DATASET = Path("/home/usr/vana/GMT2/Andrew/HEADPOSITIONCAT")

sns.set(
    font="Lato",
    font_scale=1.5,
    palette="pastel",
    style="dark",
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


def main():
    # get the two pipeline outputs
    output = DATASET / "derivatives" / "me_pipeline" / "sub-MSCHD02"
    raw_func_path = (
        DATASET / "sub-MSCHD02" / "ses-01" / "func" / "sub-MSCHD02_ses-01_task-rest_run-01_echo-1_part-mag_bold.nii.gz"
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
    medic_screenshot_path = DATASET / "medic.png"
    topup_screenshot_path = DATASET / "topup.png"

    # load data
    medic_screenshot = Image.open(medic_screenshot_path)
    topup_screenshot = Image.open(topup_screenshot_path)
    raw_func = nib.load(raw_func_path)
    topup_mag = nib.load(topup_mag_path)
    topup_mag_2 = nib.load(topup_mag_path_2)
    medic_func = nib.load(medic_func_path)
    topup_func = nib.load(topup_func_path)
    medic_fmaps = nib.load(medic_fmaps_path)
    topup_fmaps = nib.load(topup_fmaps_path)

    fontsize_1 = 14
    fontsize_2 = 12

    # create a figure
    f = plt.figure(figsize=(16, 9), layout="constrained")
    # add 2 subfigures
    subfigs = f.subfigures(2, 1, height_ratios=[1, 2], hspace=0)
    # plot screenshots in first subfigure
    sbs = subfigs[0].subplots(1, 2)
    sbs[0].imshow(topup_screenshot)
    sbs[0].set_xticks([])
    sbs[0].set_yticks([])
    sbs[0].set_title("(A) TOPUP", loc="center", y=-0.2, fontsize=fontsize_1)
    sbs[1].imshow(medic_screenshot)
    sbs[1].set_xticks([])
    sbs[1].set_yticks([])
    sbs[1].set_title("(B) MEDIC", loc="center", y=-0.2, fontsize=fontsize_1)

    # add subfigure to second subfigure
    subsubfigs = subfigs[1].subfigures(1, 2, width_ratios=[1, 4])

    # plot topup field map
    topup_fig = data_plotter(
        [topup_mag.get_fdata(), topup_mag_2.get_fdata(), topup_fmaps.get_fdata() / (2 * np.pi)],
        vmin=[topup_mag.get_fdata().min(), topup_mag_2.get_fdata().min(), -100],
        vmax=[topup_mag.get_fdata().max(), topup_mag_2.get_fdata().max(), 100],
        colormaps=["gray", "gray", "icefire"],
        figure=subsubfigs[0],
    )
    topup_fig.suptitle("(C) TOPUP Field Map", y=0.97, fontsize=fontsize_1)
    topup_axes = topup_fig.get_axes()
    topup_axes[1].set_title("PA encode", loc="center", y=-0.5, fontsize=fontsize_2)
    topup_axes[4].set_title("AP encode", loc="center", y=-0.5, fontsize=fontsize_2)
    topup_axes[7].set_title("Field map", loc="center", y=-0.5, fontsize=fontsize_2)

    # get min max
    func_min = raw_func.dataobj[..., 0].min()
    func_max = raw_func.dataobj[..., 0].max()
    fmap_min = -100
    fmap_max = 100

    # add subfigure to subsubfigure
    subsubsubfigs = subsubfigs[1].subfigures(2, 1, hspace=0.05)
    # plot medic and topup functional data in second subfigure
    subsubsubfigs[0].suptitle("(D) MEDIC Field Map", fontsize=fontsize_1)
    # make subfigure in second subfigure
    sub4xfigs_1 = subsubsubfigs[0].subfigures(2, 3, wspace=0.1)
    data_plotter(
        [medic_fmaps.dataobj[..., 50]],
        vmin=fmap_min,
        vmax=fmap_max,
        colormaps="icefire",
        figure=sub4xfigs_1[0, 0],
    ).get_axes()[1].set_title("Frame 50", loc="center", y=-0.5, fontsize=fontsize_2)
    data_plotter(
        [medic_fmaps.dataobj[..., 250]],
        vmin=fmap_min,
        vmax=fmap_max,
        colormaps="icefire",
        figure=sub4xfigs_1[0, 1],
    ).get_axes()[1].set_title("Frame 250", loc="center", y=-0.5, fontsize=fontsize_2)
    data_plotter(
        [medic_fmaps.dataobj[..., 350]],
        vmin=fmap_min,
        vmax=fmap_max,
        colormaps="icefire",
        figure=sub4xfigs_1[0, 2],
    ).get_axes()[1].set_title("Frame 350", loc="center", y=-0.5, fontsize=fontsize_2)
    data_plotter(
        [medic_fmaps.dataobj[..., 450]],
        vmin=fmap_min,
        vmax=fmap_max,
        colormaps="icefire",
        figure=sub4xfigs_1[1, 0],
    ).get_axes()[1].set_title("Frame 450", loc="center", y=-0.5, fontsize=fontsize_2)
    data_plotter(
        [medic_fmaps.dataobj[..., 550]],
        vmin=fmap_min,
        vmax=fmap_max,
        colormaps="icefire",
        figure=sub4xfigs_1[1, 1],
    ).get_axes()[1].set_title("Frame 550", loc="center", y=-0.5, fontsize=fontsize_2)
    data_plotter(
        [medic_fmaps.dataobj[..., 650]],
        vmin=fmap_min,
        vmax=fmap_max,
        colormaps="icefire",
        figure=sub4xfigs_1[1, 2],
    ).get_axes()[1].set_title("Frame 650", loc="center", y=-0.5, fontsize=fontsize_2)

    # plot associated field maps in third subfigure
    subsubsubfigs[1].suptitle("(E) Functional Data", fontsize=fontsize_1)
    # make subfigure in third subfigure
    sub4xfigs_2 = subsubsubfigs[1].subfigures(2, 3, wspace=0.1)
    data_plotter(
        [raw_func.dataobj[..., 50]],
        vmin=func_min,
        vmax=func_max,
        colormaps="gray",
        figure=sub4xfigs_2[0, 0],
    ).get_axes()[1].set_title("Frame 50", loc="center", y=-0.5, fontsize=fontsize_2)
    data_plotter(
        [raw_func.dataobj[..., 250]],
        vmin=func_min,
        vmax=func_max,
        colormaps="gray",
        figure=sub4xfigs_2[0, 1],
    ).get_axes()[1].set_title("Frame 250", loc="center", y=-0.5, fontsize=fontsize_2)
    data_plotter(
        [raw_func.dataobj[..., 350]],
        vmin=func_min,
        vmax=func_max,
        colormaps="gray",
        figure=sub4xfigs_2[0, 2],
    ).get_axes()[1].set_title("Frame 350", loc="center", y=-0.5, fontsize=fontsize_2)
    data_plotter(
        [raw_func.dataobj[..., 450]],
        vmin=func_min,
        vmax=func_max,
        colormaps="gray",
        figure=sub4xfigs_2[1, 0],
    ).get_axes()[1].set_title("Frame 450", loc="center", y=-0.5, fontsize=fontsize_2)
    data_plotter(
        [raw_func.dataobj[..., 550]],
        vmin=func_min,
        vmax=func_max,
        colormaps="gray",
        figure=sub4xfigs_2[1, 1],
    ).get_axes()[1].set_title("Frame 550", loc="center", y=-0.5, fontsize=fontsize_2)
    data_plotter(
        [raw_func.dataobj[..., 650]],
        vmin=func_min,
        vmax=func_max,
        colormaps="gray",
        figure=sub4xfigs_2[1, 2],
    ).get_axes()[1].set_title("Frame 650", loc="center", y=-0.5, fontsize=fontsize_2)

    f.savefig(FIGURE_OUT / "head_position_cat.png", dpi=300, bbox_inches="tight")

    # show plot
    plt.show()
