from bids import BIDSLayout
import nibabel as nib
from memori.logging import run_process
from memori.pathman import PathManager as PathMan
from . import (
    parser,
)
from memori.helpers import working_directory
from medic_analysis.common import data_plotter, plt, Figure

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
    # call the parser
    args = parser.parse_args()
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

    # load static and transient field map files
    medic_fieldmaps = PathMan(args.output_dir) / "fieldmaps" / "medic"
    with working_directory(medic_fieldmaps.path):
        # load static field map runs
        static_fieldmaps = []
        for idx in args.static_head_position_run_idx:
            run = idx + 1
            static_fieldmaps.append(
                nib.load(PathMan(f"run{run:02d}") / "fmap.nii.gz")
            )

        transient_fieldmaps = []
        for idx in args.transient_head_position_run_idx:
            run = idx + 1
            transient_fieldmaps.append(
                nib.load(PathMan(f"run{run:02d}") / "fmap.nii.gz")
            )
    
    breakpoint()
