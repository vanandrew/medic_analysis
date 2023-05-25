from bids import BIDSLayout
import simplebrainviewer as sbv
import nibabel as nib
import numpy as np
from memori.pathman import PathManager as PathMan
from memori.helpers import working_directory
from memori.logging import setup_logging
from warpkit.utilities import field_maps_to_displacement_maps, displacement_map_to_field, resample_image
import subprocess
from . import (
    parser,
    PED_TABLE,
    POLARITY_IDX,
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

STATIC_HEAD_POSITION_RUN_IDX = [0, 1, 2, 3, 4, 5, 6]
TOTAL_READOUT_TIME = 0.0305196


def main():
    # call the parser
    args = parser.parse_args()
    setup_logging()

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

    # load each functional image (1st frame)
    funcs = []
    fmap_mats = []
    func_mats = []
    for run_idx in STATIC_HEAD_POSITION_RUN_IDX:
        funcs.append(layout.get(suffix="bold", part="mag", run=run_idx + 1, echo=2, extension="nii.gz")[0])
        # get the transformation matrix for each field map run
        fmap_mats.append(
            output_dir / "framewise_align" / "fmap" / f"run{run_idx+1:02d}" / f"run{run_idx+1:02d}.mat" / "MAT_0000"
        )
        # get the transformation matrix for each func run
        func_mats.append(
            output_dir / "framewise_align" / "func" / f"run{run_idx+1:02d}" / f"run{run_idx+1:02d}.mat" / "MAT_0000"
        )

    # grab the 1st frame from each run
    funcs = [
        nib.Nifti1Image(func.get_image().dataobj[..., 0], func.get_image().affine, func.get_image().header)
        for func in funcs
    ]

    # load the neutral run field map displacment field
    fmap = output_dir / "fieldmaps" / "topup" / "run01" / "fout.nii.gz"

    # get medic aligned/corrected images
    medic_funcs = [
        output_dir / "corrected_data" / f"run{run_idx+1:02d}" / "medic" / "medic_corrected_aligned.nii.gz"
        for run_idx in STATIC_HEAD_POSITION_RUN_IDX
    ]

    # cross apply field maps
    cross_align_dir = PathMan(output_dir / "cross_align")
    cross_align_dir.mkdir(exist_ok=True)
    with working_directory(cross_align_dir):
        for run_idx in STATIC_HEAD_POSITION_RUN_IDX:
            # invert the fmap affine
            subprocess.run(
                [
                    "convert_xfm",
                    "-omat",
                    f"fmap{run_idx+1:02d}_inv.mat",
                    "-inverse",
                    fmap_mats[run_idx],
                ],
                stdout=True,
                stderr=True,
                check=True,
            )

            # apply the transform to the fmap
            subprocess.run(
                [
                    "flirt",
                    "-in",
                    fmap.path,
                    "-ref",
                    fmap.path,
                    "-applyxfm",
                    "-init",
                    f"fmap{run_idx+1:02d}_inv.mat",
                    "-out",
                    f"fmap{run_idx+1:02d}.nii.gz",
                    "-interp",
                    "spline",
                    "-v",
                ],
                stdout=True,
                stderr=True,
                check=True,
            )

            xfmd_fmap = nib.load(f"fmap{run_idx+1:02d}.nii.gz")

            # add 4th dim
            xfmd_fmap = nib.Nifti1Image(
                xfmd_fmap.get_fdata()[..., np.newaxis],
                xfmd_fmap.affine,
                xfmd_fmap.header,
            )

            # convert the field map to a displacement field
            dmap = field_maps_to_displacement_maps(
                xfmd_fmap,
                TOTAL_READOUT_TIME,
                "y-",
            )
            dfield = displacement_map_to_field(dmap, "y-", format="fsl", frame=0)
            dfield.to_filename(f"dfield{run_idx+1:02d}.nii.gz")

            # resample image
            resampled_func = resample_image(
                funcs[run_idx],
                funcs[run_idx],
                dfield,
            )
            resampled_func.to_filename(f"func{run_idx+1:02d}.nii.gz")

            # now align the resampled func back to the neutral position
            subprocess.run(
                [
                    "flirt",
                    "-in",
                    f"func{run_idx+1:02d}.nii.gz",
                    "-ref",
                    f"func{run_idx+1:02d}.nii.gz",
                    "-out",
                    f"topup{run_idx+1:02d}.nii.gz",
                    "-applyxfm",
                    "-init",
                    func_mats[run_idx],
                    "-interp",
                    "sinc",
                    "-v",
                ],
                stdout=True,
                stderr=True,
                check=True,
            )

            # get the medic equivalent frame
            medic_func = nib.load(medic_funcs[run_idx])
            nib.Nifti1Image(medic_func.dataobj[..., 0], medic_func.affine, medic_func.header).to_filename(
                f"medic{run_idx+1:02d}.nii.gz"
            )
