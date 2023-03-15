import nibabel as nib
from bids import BIDSLayout
from memori.pathman import PathManager as PathMan
from . import (
    parser,
)
from medic_analysis.common import data_plotter, render_dynamic_figure, plt

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
        # load static field map runs
        static_fieldmaps = []
        for idx in args.static_head_position_run_idx:
            run = idx + 1
            static_fieldmaps.append(nib.load(medic_fieldmaps / f"run{run:02d}" / "fmap.nii.gz"))

        # plot static field maps
        f0 = plt.figure(figsize=(16, 8), layout="constrained")
        f0_subfigs = f0.subfigures(1, 2)
        data_plotter(
            [fmap for fmap in static_fieldmaps[:4]],
            figure=f0_subfigs[0],
            frame_num=20,
            colorbar=True,
            colorbar_alt_range=True,
        )
        f0_subfigs[0].text(0.6, 0.76, f"(A) {args.labels[0]}", ha="center")
        f0_subfigs[0].text(0.6, 0.51, f"(B) {args.labels[1]}", ha="center")
        f0_subfigs[0].text(0.6, 0.26, f"(C) {args.labels[2]}", ha="center")
        f0_subfigs[0].text(0.6, 0.01, f"(D) {args.labels[3]}", ha="center")

        data_plotter(
            [fmap for fmap in static_fieldmaps[4:]],
            figure=f0_subfigs[1],
            frame_num=20,
            colorbar2=True,
            colorbar2_alt_range=True,
        )
        f0_subfigs[1].text(0.4, 0.76, f"(E) {args.labels[4]}", ha="center")
        f0_subfigs[1].text(0.4, 0.51, f"(F) {args.labels[5]}", ha="center")
        f0_subfigs[1].text(0.4, 0.26, f"(G) {args.labels[6]}", ha="center")
        f0_subfigs[1].text(0.4, 0.01, f"(H) {args.labels[14]}", ha="center")

    if args.plot_only is None or 1 in args.plot_only:
        # load up corrected images
        medic_corrected = PathMan(args.output_dir) / "corrected_data"
        corrected_data_medic = []
        for idx in args.static_head_position_run_idx:
            run = idx + 1
            corrected_data_medic.append(
                nib.load(medic_corrected / f"run{run:02d}" / "medic" / "medic_corrected.nii.gz")
            )
        topup_corrected = PathMan(args.output_dir) / "corrected_data"
        corrected_data_topup = []
        for idx in args.static_head_position_run_idx:
            run = idx + 1
            corrected_data_topup.append(
                nib.load(topup_corrected / f"run{run:02d}" / "topup" / "topup_corrected.nii.gz")
            )

        # compute tSNR for first static field map
        tSNR_difference = []
        for medic, topup in zip(corrected_data_medic, corrected_data_topup):
            tSNR_medic = medic.get_fdata().mean(axis=3) / medic.get_fdata().std(axis=3)
            tSNR_topup = topup.get_fdata().mean(axis=3) / topup.get_fdata().std(axis=3)
            tSNR_difference.append(tSNR_medic - tSNR_topup)
        # plot tSNR
        f1 = plt.figure(figsize=(16, 8), layout="constrained")
        f1_subfigs = f1.subfigures(1, 2)
        data_plotter(
            tSNR_difference[:4],
            figure=f1_subfigs[0],
            colorbar=True,
            colorbar_label="tSNR Difference",
            vmax=10,
            vmin=-10,
        )
        f1_subfigs[0].text(0.6, 0.76, f"(A) {args.labels[0]}", ha="center")
        f1_subfigs[0].text(0.6, 0.51, f"(B) {args.labels[1]}", ha="center")
        f1_subfigs[0].text(0.6, 0.26, f"(C) {args.labels[2]}", ha="center")
        f1_subfigs[0].text(0.6, 0.01, f"(D) {args.labels[3]}", ha="center")
        data_plotter(
            tSNR_difference[:4],
            figure=f1_subfigs[1],
            colorbar2=True,
            colorbar2_label="tSNR Difference",
            vmax=10,
            vmin=-10,
        )
        f1_subfigs[1].text(0.4, 0.76, f"(E) {args.labels[4]}", ha="center")
        f1_subfigs[1].text(0.4, 0.51, f"(F) {args.labels[5]}", ha="center")
        f1_subfigs[1].text(0.4, 0.26, f"(G) {args.labels[6]}", ha="center")
        f1_subfigs[1].text(0.4, 0.01, f"(H) {args.labels[14]}", ha="center")

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
        transients_out = (PathMan(args.output_dir) / "transients")
        transients_out.mkdir(exist_ok=True)

        # render transient field map videos
        for fmap, label in zip(transient_fieldmaps, labels):
            render_dynamic_figure(
                str(transients_out / f"{label}.mp4"),
                [fmap],
                colorbar=True,
                colorbar_alt_range=True,
            )

    # plot static field maps
    plt.show()
