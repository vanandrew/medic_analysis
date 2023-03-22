from memori.pathman import PathManager as PathMan
from memori.helpers import working_directory
from bids import BIDSLayout
import numpy as np
import nibabel as nib
from warpkit.distortion import medic
from medic_analysis.common import framewise_align
from . import parser

# Define the path to the BIDS dataset
BIDS_DATA_DIR = "/home/usr/vana/GMT2/Andrew/SLICETEST"


def main():

    # Create additional argument for multiprocessing
    parser.add_argument(
        "--multiproc",
        type=bool,
        help="Boolean flag to use multiprocessing. Defaults to False.",
        default=False,
        )

    # call the parser
    args = parser.parse_args()

    # determine number of cpus based on multiprocessing flag
    n_cpus = 8 if args.multiproc else 1

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

    # loop over runs
    runs = layout.get_runs(datatype="func")

    for run in runs:
        (output_dir / f"run-{run}").mkdir(exist_ok=True)
        with working_directory((output_dir / f"run-{run}").path):
            mag = layout.get(part="mag", datatype="func", run=run, extension="nii.gz")
            phase = layout.get(part="phase", datatype="func", run=run, extension="nii.gz")
            mag_imgs = [m.get_image() for m in mag]
            phase_imgs = [p.get_image() for p in phase]
            TEs = [m.get_metadata()["EchoTime"] * 1000 for m in mag]
            total_readout_time = mag[0].get_metadata()["TotalReadoutTime"]
            phase_encoding_direction = mag[0].get_metadata()["PhaseEncodingDirection"]

            # get reference frame
            ref_data = mag_imgs[0].dataobj[..., 0]
            nib.Nifti1Image(ref_data, mag_imgs[0].affine).to_filename("ref.nii")

            # compute motion parameters
            framewise_align(
                "ref.nii",
                mag[0].path,
                "mcflirt"
            )

            # load in motion params, convert rotations to mm
            motion_params = np.loadtxt("mcflirt.par")
            motion_params[:, :3] = np.rad2deg(motion_params[:, :3])
            motion_params[:, :3] = 50 * (np.pi / 180) * motion_params[:, :3]

            fmap_native, dmap, fmap = medic(
                phase_imgs,
                mag_imgs,
                TEs,
                total_readout_time,
                phase_encoding_direction,
                n_cpus=n_cpus,
                motion_params=motion_params,
                frames=list(range(510))
            )
            fmap_native.to_filename("fmap_native.nii.gz")
            dmap.to_filename("dmap.nii.gz")
            fmap.to_filename("fmap.nii.gz")
