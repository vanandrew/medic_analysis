import logging
import nibabel as nib
import numpy as np
from bids import BIDSLayout
from memori.pathman import PathManager as PathMan
from memori.logging import run_process
from memori.helpers import working_directory
from warpkit.utilities import (
    field_maps_to_displacement_maps,
    invert_displacement_maps,
    displacement_maps_to_field_maps,
    displacement_map_to_field,
    resample_image,
)
from warpkit.model import fit_motion_model, apply_motion_model
from medic_analysis.common import (
    apply_framewise_mats,
)
from . import parser

# Define the path to the BIDS dataset
BIDS_DATA_DIR = "/home/usr/vana/GMT2/Andrew/HEADPOSITIONSUSTEST"


def main():
    # call the parser
    args = parser.parse_args()

    # if bids dir not specified, use default
    if args.bids_dir is None:
        args.bids_dir = BIDS_DATA_DIR

    # Load the dataset
    layout = BIDSLayout(args.bids_dir, database_path=args.bids_dir)

    # determine number of echoes in data
    n_echos = len(layout.get(datatype="func", extension="nii.gz", suffix="bold", part="mag", run="01"))

    # Get the phase
    me_epi_mag_data = [
        echoes
        for echoes in zip(
            *[
                layout.get(datatype="func", extension="nii.gz", echo=echo, part="mag", suffix="bold")
                for echo in range(1, n_echos + 1)
            ]
        )
    ]
    me_epi_phase_data = [
        echoes
        for echoes in zip(
            *[
                layout.get(datatype="func", extension="nii.gz", echo=echo, part="phase", suffix="bold")
                for echo in range(1, n_echos + 1)
            ]
        )
    ]

    # get metadata
    total_readout_time = me_epi_phase_data[0][0].get_metadata()["TotalReadoutTime"]
    phase_encoding_direction = me_epi_phase_data[0][0].get_metadata()["PhaseEncodingDirection"]

    # set output dir
    if args.output_dir is None:
        args.output_dir = (PathMan(args.bids_dir) / "derivatives").path

    # make sure the output dir exists
    if not PathMan(args.output_dir).exists():
        raise FileExistsError(f"Output directory {args.output_dir} does not exist.")

    # load up all the medic field maps in native space
    medic_fmaps_output = PathMan(args.output_dir) / "fieldmaps" / "medic_aligned"
    framewise_align_output = PathMan(args.output_dir) / "framewise_align" / "func"
    medic_fmaps = []
    motion_params = []
    for run in range(1, 16):
        logging.info("Loading run %d" % run)
        # load fmap
        fmap_img = nib.load(medic_fmaps_output / f"run{run:02d}" / "fmap_native.nii.gz")

        # restrict to first 100 frames
        fmap_data = fmap_img.dataobj[..., :100]

        # make new image and append to list
        medic_fmaps.append(nib.Nifti1Image(fmap_data, fmap_img.affine, fmap_img.header))

        # load the motion parameters and append to list
        motion_params.append(np.loadtxt(framewise_align_output / f"run{run:02d}" / f"run{run:02d}.par")[:100, :])

    # fit the motion model
    logging.info("Fitting motion model")
    weights = fit_motion_model(medic_fmaps, motion_params)

    # get the model stabilized field maps
    ms_output = PathMan(args.output_dir) / "super_linear_model"
    ms_output.mkdir(exist_ok=True)
    for i, run in enumerate(range(1, 16)):
        logging.info("Stabilizing run %d" % run)
        # apply the motion model
        stabilized_fmap_native_ref_data = apply_motion_model(weights, motion_params[i], i)

        # make into image
        stabilized_fmap_native_ref = nib.Nifti1Image(
            stabilized_fmap_native_ref_data, medic_fmaps[i].affine, medic_fmaps[i].header
        )
        (ms_output / f"run{run:02d}").mkdir(exist_ok=True)
        with working_directory(ms_output / f"run{run:02d}"):
            stabilized_fmap_native_ref.to_filename("fmap_native_refaligned.nii.gz")
            # realign into native space
            for mat in (framewise_align_output / f"run{run:02d}" / f"run{run:02d}.mat").glob("MAT*"):
                run_process(
                    [
                        "convert_xfm",
                        "-omat",
                        mat.parent / f"INV{mat.name}",
                        "-inverse",
                        mat.path,
                    ]
                )
            # apply the inverse transforms
            apply_framewise_mats(
                "fmap_native_refaligned.nii.gz",
                "fmap_native_refaligned.nii.gz",
                (framewise_align_output / f"run{run:02d}" / f"run{run:02d}.mat").path,
                out_prefix="fmap_native",
                mat_prefix="INVMAT_",
            )
        # load the stabilized field map
        stabilized_fmap_native = nib.load(ms_output / f"run{run:02d}" / "fmap_native.nii.gz")

        # now convert to displacement map
        inv_displacement_maps = field_maps_to_displacement_maps(
            stabilized_fmap_native, total_readout_time, phase_encoding_direction
        )
        # invert
        displacement_maps = invert_displacement_maps(inv_displacement_maps, phase_encoding_direction)

        # convert back to field map
        stabilized_fmap = displacement_maps_to_field_maps(
            displacement_maps,
            total_readout_time,
            phase_encoding_direction,
            flip_sign=True,
        )

        # save images
        with working_directory(ms_output / f"run{run:02d}"):
            stabilized_fmap_native.to_filename("fmap_native.nii.gz")
            displacement_maps.to_filename("displacement_maps.nii.gz")
            stabilized_fmap.to_filename("fmap.nii.gz")

            # apply corrections to data
            logging.info("Applying corrections to data")
            # get the first echo mag data
            first_echo_mag_img = me_epi_mag_data[i][0].get_image()
            corrected_data = np.zeros_like(first_echo_mag_img.dataobj[..., :100])
            for frame_idx in range(100):
                logging.info(f"Applying correction to frame {frame_idx}")
                # get the frame
                frame = first_echo_mag_img.dataobj[..., frame_idx]
                frame_img = nib.Nifti1Image(frame, first_echo_mag_img.affine, first_echo_mag_img.header)

                # now get the displacment field
                displacement_map = nib.Nifti1Image(
                    displacement_maps.dataobj[..., frame_idx], displacement_maps.affine, displacement_maps.header
                )
                displacement_field = displacement_map_to_field(displacement_map, phase_encoding_direction)

                # resample frame
                corrected_frame = resample_image(
                    frame_img,
                    frame_img,
                    displacement_field,
                )
                corrected_data[..., frame_idx] = corrected_frame.get_fdata()

            # save corrected data
            corrected_img = nib.Nifti1Image(corrected_data, first_echo_mag_img.affine, first_echo_mag_img.header)
            corrected_img.to_filename("corrected_data.nii.gz")
