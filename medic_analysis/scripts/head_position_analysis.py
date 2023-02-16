import nibabel as nib
from bids import BIDSLayout
from memori.pathman import PathManager as PathMan
from memori.helpers import working_directory
from medic_analysis.common.align import framewise_align
from . import parser

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
    "High Resolution SE PEPolar Field Map",
]

# Define constants for run numbers and indices
STATIC_HEAD_POSITION_RUN_NUMBER = [1, 2, 3, 4, 5, 6, 7, 15]
STATIC_HEAD_POSITION_RUN_IDX = [0, 1, 2, 3, 4, 5, 6, 14]
TRANSIENT_HEAD_POSITION_RUN_NUMBER = [8, 9, 10, 11, 12, 13, 14]
TRANSIENT_HEAD_POSITION_RUN_IDX = [7, 8, 9, 10, 11, 12, 13]

# Set polarity index
POLARITY_IDX = {"PA": 0, "AP": 1}

# Define the path to the BIDS dataset
BIDS_DATA_DIR = "/home/usr/vana/GMT2/Andrew/HEADPOSITIONSUSTEST"


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
        "--ref_frame_run_number",
        type=int,
        help="Run number to extract reference frame from.",
        default=0,
    )
    parser.add_argument(
        "--ref_frame_idx",
        type=int,
        help="Index of the reference frame for the head position analysis.",
        default=0,
    )
    parser.add_argument(
        "--ref_fmap_polarity",
        help="Polarity of the reference field map. Options are 'AP' or 'PA'. By default PA.",
        default="PA",
        choices=["AP", "PA"],
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

    # make the output dir if not exist
    output_dir = PathMan(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # determine number of echoes in data
    n_echos = len(layout.get(datatype="func", extension="nii.gz", suffix="bold", part="mag", run="01"))

    # Get the ME-EPI data for each of the runs
    # Get the magnitude
    me_epi_mag_data = [
        echoes
        for echoes in zip(
            *[
                layout.get(datatype="func", extension="nii.gz", echo=echo, part="mag", suffix="bold")
                for echo in range(1, n_echos + 1)
            ]
        )
    ]
    # Get the phase
    me_epi_phase_data = [
        echoes
        for echoes in zip(
            *[
                layout.get(datatype="func", extension="nii.gz", echo=echo, part="phase", suffix="bold")
                for echo in range(1, n_echos + 1)
            ]
        )
    ]

    # Get the PE-Polar field maps for each of the runs
    # Make tuples of AP and PA maps
    pepolar_fmap_data = [
        (pa, ap)
        for pa, ap in zip(
            layout.get(datatype="fmap", direction="PA", extension="nii.gz", suffix="epi"),
            layout.get(datatype="fmap", direction="AP", extension="nii.gz", suffix="epi"),
        )
    ]

    # Now split the data into static and transient
    static_me_epi_mag_data = [me_epi_mag_data[idx] for idx in STATIC_HEAD_POSITION_RUN_IDX]
    static_me_epi_phase_data = [me_epi_phase_data[idx] for idx in STATIC_HEAD_POSITION_RUN_IDX]
    static_pepolar_fmap_data = [pepolar_fmap_data[idx] for idx in STATIC_HEAD_POSITION_RUN_IDX]
    static_labels = [RUN_LABELS[idx] for idx in STATIC_HEAD_POSITION_RUN_IDX]

    transient_me_epi_mag_data = [me_epi_mag_data[idx] for idx in TRANSIENT_HEAD_POSITION_RUN_IDX]
    transient_me_epi_phase_data = [me_epi_phase_data[idx] for idx in TRANSIENT_HEAD_POSITION_RUN_IDX]
    transient_pepolar_fmap_data = [pepolar_fmap_data[idx] for idx in TRANSIENT_HEAD_POSITION_RUN_IDX]
    transient_labels = [RUN_LABELS[idx] for idx in TRANSIENT_HEAD_POSITION_RUN_IDX]

    # extract the reference frame for the me and pepolar data
    # for me-epi data use the first echo
    # for pepolar data grab, the correct polarity
    me_epi_ref_run = me_epi_mag_data[args.ref_frame_idx][0].get_image()
    pepolar_fmap_ref_run = pepolar_fmap_data[args.ref_frame_idx][POLARITY_IDX[args.ref_fmap_polarity]].get_image()
    # get the frame
    # for pepolar data, average across all frames
    me_epi_ref_frame = me_epi_ref_run.get_fdata()[..., args.ref_frame_idx]
    pepolar_fmap_ref_frame = pepolar_fmap_ref_run.get_fdata().mean(axis=-1)

    # switch to output dir
    (output_dir / "references").mkdir(exist_ok=True)
    with working_directory(str(output_dir / "references")):
        # now write out the reference frames to a new file
        nib.Nifti1Image(me_epi_ref_frame, me_epi_ref_run.affine).to_filename("me_epi_ref.nii.gz")

        # now write out the fmap reference
        nib.Nifti1Image(pepolar_fmap_ref_frame, pepolar_fmap_ref_run.affine).to_filename("pepolar_fmap_ref.nii.gz")

    # set path to references
    me_epi_ref_path = str(output_dir / "references" / "me_epi_ref.nii.gz")
    pepolar_fmap_ref_path = str(output_dir / "references" / "pepolar_fmap_ref.nii.gz")

    # create framewise_align directory
    (output_dir / "framewise_align").mkdir(exist_ok=True)
    with working_directory(str(output_dir / "framewise_align")):
        run = 1
        PathMan(f"run{run:02d}").mkdir(exist_ok=True)

        # now run framewise alignment
        framewise_align(
            me_epi_ref_path,
            me_epi_mag_data[0][0].path,
            (PathMan(f"run{run:02d}") / f"run{run:02d}").path
        )
