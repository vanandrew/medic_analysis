"""Perform head position analysis.

I think the figure is Figure 1, but the actual annotations don't match what's in the preprint.
"""
import nibabel as nib
import numpy as np
from bids import BIDSLayout
from memori.pathman import PathManager as PathMan
from warpkit.unwrap import create_brain_mask

from medic_analysis.common.figures import FIGURE_OUT, data_plotter, plt, render_dynamic_figure, sns

from . import parser

sns.set(
    font="Lato",
    font_scale=1.25,
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

# Define the path to the BIDS dataset
BIDS_DATA_DIR = "/home/usr/vana/GMT2/Andrew/HEADPOSITIONSUSTEST"

# create a list of labels for each run
RUN_LABELS = [
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

# Define constants for run and indices
STATIC_HEAD_POSITION_RUN_IDX = [0, 1, 2, 3, 4, 5, 6, 14]
TRANSIENT_HEAD_POSITION_RUN_IDX = [7, 8, 9, 10, 11, 12, 13]


def main():
    # add arguments to parser
    parser.add_argument("--labels", help="List of labels for each run.", nargs="+", default=RUN_LABELS)
    parser.add_argument(
        "--static_head_position_run_idx",
        nargs="+",
        type=int,
        help="List of run indices for static head position runs.",
        default=STATIC_HEAD_POSITION_RUN_IDX,
    )
    parser.add_argument(
        "--transient_head_position_run_idx",
        nargs="+",
        type=int,
        help="List of run indices for transient head position runs.",
        default=TRANSIENT_HEAD_POSITION_RUN_IDX,
    )
    parser.add_argument(
        "--plot_only",
        nargs="+",
        type=int,
        help="List of figures to plot.",
        default=None,
    )

    # call the parser
    args = parser.parse_args()

    # if bids dir not specified, use default
    if args.bids_dir is None:
        args.bids_dir = BIDS_DATA_DIR

    # Load the dataset
    layout = BIDSLayout(args.bids_dir, database_path=args.bids_dir)

    # set output dir
    if args.output_dir is None:
        args.output_dir = (PathMan(args.bids_dir) / "derivatives").path

    # make sure the output dir exists
    if not PathMan(args.output_dir).exists():
        raise FileExistsError(f"Output directory {args.output_dir} does not exist.")

    if args.plot_only is None or 0 in args.plot_only:
        # load field map files
        medic_fieldmaps = PathMan(args.output_dir) / "fieldmaps" / "medic_aligned"
        # load topup field map in neutral position as reference
        topup_fieldmap = nib.load(
            PathMan(args.output_dir) / "fieldmaps" / "topup" / "run01" / "fout.nii.gz"
        ).get_fdata()
        # load static field map runs
        static_fieldmaps = []
        for idx in args.static_head_position_run_idx:
            run = idx + 1
            static_fieldmaps.append(nib.load(medic_fieldmaps / f"run{run:02d}" / "fmap.nii.gz").dataobj)
        # load mask
        mask = nib.load(PathMan(args.output_dir) / "references" / "me_epi_ref_bet_mask.nii.gz").get_fdata()

        # plot range
        vlims = (-50, 50)

        # plot static field maps
        f0 = plt.figure(figsize=(16, 8), layout="constrained")
        f0_subfigs = f0.subfigures(1, 2)
        data_plotter(
            [
                (static_fieldmaps[1][..., 0] - static_fieldmaps[0][..., 0]) * mask,
                (static_fieldmaps[3][..., 0] - static_fieldmaps[0][..., 0]) * mask,
                (static_fieldmaps[5][..., 0] - static_fieldmaps[0][..., 0]) * mask,
            ],
            figure=f0_subfigs[0],
            colorbar=True,
            colorbar_alt_range=True,
            vmin=vlims[0],
            vmax=vlims[1],
            text_color="white",
        )
        sbs = f0_subfigs[0].get_axes()
        sbs[1].set_title(f"(A) {args.labels[1]} (14.957 deg)", loc="center", y=-0.5)
        sbs[4].set_title(f"(C) {args.labels[3]} (10.642 deg)", loc="center", y=-0.5)
        sbs[7].set_title(f"(E) {args.labels[5]} (10.789 deg)", loc="center", y=-0.5)

        data_plotter(
            [
                (static_fieldmaps[2][..., 0] - static_fieldmaps[0][..., 0]) * mask,
                (static_fieldmaps[4][..., 0] - static_fieldmaps[0][..., 0]) * mask,
                (static_fieldmaps[6][..., 0] - static_fieldmaps[0][..., 0]) * mask,
            ],
            figure=f0_subfigs[1],
            colorbar2=True,
            colorbar2_alt_range=True,
            vmin=vlims[0],
            vmax=vlims[1],
            text_color="white",
        )
        sbs = f0_subfigs[1].get_axes()
        sbs[1].set_title(f"(B) {args.labels[2]} (9.778 deg)", loc="center", y=-0.5)
        sbs[4].set_title(f"(D) {args.labels[4]} (13.726 deg)", loc="center", y=-0.5)
        sbs[7].set_title(f"(F) {args.labels[6]} (8.577 deg)", loc="center", y=-0.5)
        f0.suptitle("Motion-dependent field map differences (Position - Neutral Position)")
        f0.savefig(FIGURE_OUT / "fieldmap_differences.png", dpi=300, bbox_inches="tight")

        # plot static field maps against topup neutral
        f1 = plt.figure(figsize=(16, 8), layout="constrained")
        f1_subfigs = f1.subfigures(1, 2)
        data_plotter(
            [
                (static_fieldmaps[0][..., 0] - topup_fieldmap) * mask,
                (static_fieldmaps[1][..., 0] - topup_fieldmap) * mask,
                (static_fieldmaps[3][..., 0] - topup_fieldmap) * mask,
                (static_fieldmaps[5][..., 0] - topup_fieldmap) * mask,
            ],
            figure=f1_subfigs[0],
            colorbar=True,
            colorbar_alt_range=True,
            vmin=vlims[0],
            vmax=vlims[1],
            text_color="white",
        )
        sbs = f1_subfigs[0].get_axes()
        sbs[1].set_title(f"(A) {args.labels[0]}", loc="center", y=-0.3)
        sbs[4].set_title(f"(B) {args.labels[1]} (14.957 deg)", loc="center", y=-0.3)
        sbs[7].set_title(f"(D) {args.labels[3]} (10.642 deg)", loc="center", y=-0.3)
        sbs[10].set_title(f"(F) {args.labels[5]} (10.789 deg)", loc="center", y=-0.3)

        data_plotter(
            [
                np.zeros_like(topup_fieldmap),
                (static_fieldmaps[2][..., 0] - topup_fieldmap) * mask,
                (static_fieldmaps[4][..., 0] - topup_fieldmap) * mask,
                (static_fieldmaps[6][..., 0] - topup_fieldmap) * mask,
            ],
            figure=f1_subfigs[1],
            colormaps=["gray", "icefire", "icefire", "icefire"],
            colorbar2=True,
            colorbar2_alt_range=True,
            colorbar2_source_idx=(1, 1),
            vmin=[0, vlims[0], vlims[0], vlims[0]],
            vmax=[10000, vlims[1], vlims[1], vlims[1]],
            text_color="white",
        )
        sbs = f1_subfigs[1].get_axes()
        sbs[0].set_visible(False)
        sbs[2].set_visible(False)
        sbs[1].set_title(f"X", loc="center", y=-0.3, color="black")
        sbs[4].set_title(f"(C) {args.labels[2]} (9.778 deg)", loc="center", y=-0.3)
        sbs[7].set_title(f"(E) {args.labels[4]} (13.726 deg)", loc="center", y=-0.3)
        sbs[10].set_title(f"(G) {args.labels[6]} (8.577 deg)", loc="center", y=-0.3)
        f1.suptitle("Field map difference from neutral position (MEDIC - TOPUP Neutral)")
        f1.savefig(FIGURE_OUT / "fieldmap_differences_topup.png", dpi=300, bbox_inches="tight")

    if args.plot_only is None or 1 in args.plot_only:
        # load up corrected images
        medic_corrected = PathMan(args.output_dir) / "corrected_data"
        corrected_data_medic = []
        for idx in args.static_head_position_run_idx:
            run = idx + 1
            corrected_data_medic.append(
                nib.load(medic_corrected / f"run{run:02d}" / "medic" / "medic_corrected_aligned.nii.gz")
            )
        topup_corrected = PathMan(args.output_dir) / "corrected_data"
        corrected_data_topup = []
        for idx in args.static_head_position_run_idx:
            run = idx + 1
            corrected_data_topup.append(
                nib.load(topup_corrected / f"run{run:02d}" / "topup" / "topup_aligned_corrected.nii.gz")
            )

        # compute tSNR for first static field map
        tSNR_medic = []
        brain_mask_medic = []
        tSNR_topup = []
        brain_mask_topup = []
        tSNR_difference = []
        for medic, topup in zip(corrected_data_medic, corrected_data_topup):
            tSNR_medic.append(medic.get_fdata().mean(axis=3) / medic.get_fdata().std(axis=3))
            brain_mask_medic.append(create_brain_mask(medic.get_fdata().mean(axis=3), 0))
            tSNR_topup.append(topup.get_fdata().mean(axis=3) / topup.get_fdata().std(axis=3))
            brain_mask_topup.append(create_brain_mask(topup.get_fdata().mean(axis=3), 0))
            tSNR_difference.append(tSNR_medic[-1] - tSNR_topup[-1])
            # plot tSNR
            f1 = plt.figure(figsize=(16, 8), layout="constrained")
            data_plotter(
                [tSNR_medic[-1], tSNR_topup[-1]],
                colorbar=True,
                colorbar_label="tSNR",
                vmax=50,
                vmin=0,
                figure=f1,
                text_color="white",
            )
            f1.text(0.6, 0.51, f"(A) MEDIC", ha="center")
            f1.text(0.6, 0.01, f"(B) TOPUP", ha="center")
            break

    if args.plot_only is None or 2 in args.plot_only:
        # load field map files
        medic_fieldmaps = PathMan(args.output_dir) / "fieldmaps" / "medic_aligned"
        # load transient field map runs
        transient_fieldmaps = []
        for idx in args.transient_head_position_run_idx:
            run = idx + 1
            transient_fieldmaps.append(nib.load(medic_fieldmaps / f"run{run:02d}" / "fmap.nii.gz"))

        # get labels
        labels = [args.labels[i] for i in args.transient_head_position_run_idx]
        # replace space with underscores
        labels = [label.replace(" ", "_") for label in labels]

        # make transients output directory
        transients_out = PathMan(args.output_dir) / "transients"
        transients_out.mkdir(exist_ok=True)

        # load motion parameters
        motion_params = []
        for idx in args.transient_head_position_run_idx:
            run = idx + 1
            motion_params.append(
                np.loadtxt(
                    PathMan(args.output_dir) / "framewise_align" / "func" / f"run{run:02d}" / f"run{run:02d}.par"
                )
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

        for fmap, moco, label in zip(transient_fieldmaps, motion_params, labels):
            render_dynamic_figure(
                str(transients_out / f"{label}.mp4"),
                [fmap],
                colorbar=True,
                colorbar_alt_range=True,
                figure_fx=set_moco_label(moco),
            )

    # plot static field maps
    # plt.show()
