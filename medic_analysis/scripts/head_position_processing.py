import logging
import nibabel as nib
import numpy as np
from bids import BIDSLayout
from memori.stage import Stage
from memori.pathman import PathManager as PathMan
from memori.helpers import working_directory
from medic_analysis.common import (
    apply_framewise_mats,
    run_topup,
    framewise_align,
    run_medic,
)
from warpkit.utilities import (
    displacement_map_to_field,
    resample_image,
    convert_warp,
)
from . import (
    parser,
    PED_TABLE,
    POLARITY_IDX,
)


# Define the path to the BIDS dataset
BIDS_DATA_DIR = "/home/usr/vana/GMT2/Andrew/HEADPOSITIONSUSTEST"


def main():
    # add arguments to parser
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
    parser.add_argument(
        "--num_frames",
        type=int,
        default=100,
        help="Number of frames to use for the head position analysis.",
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

    # get total number of rest runs in data
    n_runs = len(layout.get_runs(datatype="func"))
    assert n_runs == len(layout.get_runs(datatype="fmap")), "fmaps should have equal number of runs as func"

    # determine number of echoes in data
    n_echos = len(layout.get(datatype="func", extension="nii.gz", suffix="bold", part="mag", run="01"))

    # Get the ME-EPI data for each of the runs
    # Get the magnitude
    me_epi_mag_data = [
        list(echoes)
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

    # get the func polarity
    func_pe = POLARITY_IDX[args.ref_fmap_polarity]

    # extract the reference frame for the me and pepolar data
    # for me-epi data use the first echo
    # for pepolar data grab, the correct polarity
    me_epi_ref_run = me_epi_mag_data[args.ref_frame_idx][0].get_image()
    pepolar_fmap_ref_run = pepolar_fmap_data[args.ref_frame_idx][func_pe].get_image()
    # get the frame
    # for pepolar data, average across all frames
    me_epi_ref_frame = me_epi_ref_run.get_fdata()[..., args.ref_frame_idx]
    pepolar_fmap_ref_frame = pepolar_fmap_ref_run.get_fdata().mean(axis=-1)

    # make directory for hash outputs
    hash_outputs = output_dir / "hash_outputs"
    hash_outputs.mkdir(exist_ok=True)

    # switch to output dir
    (output_dir / "references").mkdir(exist_ok=True)
    with working_directory(str(output_dir / "references")):
        # now write out the reference frames to a new file
        if not PathMan("me_epi_ref.nii.gz").exists():
            nib.Nifti1Image(me_epi_ref_frame, me_epi_ref_run.affine).to_filename("me_epi_ref.nii.gz")

        # now write out the fmap reference
        if not PathMan("pepolar_fmap_ref.nii.gz").exists():
            nib.Nifti1Image(pepolar_fmap_ref_frame, pepolar_fmap_ref_run.affine).to_filename("pepolar_fmap_ref.nii.gz")

    # set path to references
    me_epi_ref_path = str(output_dir / "references" / "me_epi_ref.nii.gz")
    pepolar_fmap_ref_path = str(output_dir / "references" / "pepolar_fmap_ref.nii.gz")

    # create framewise_align directory
    func_aligned_output = output_dir / "framewise_align" / "func"
    func_aligned_output.mkdir(exist_ok=True, parents=True)
    with working_directory(func_aligned_output.path):
        for idx, run in zip(range(n_runs), layout.get_runs(datatype="func")):
            PathMan(f"run{run:02d}").mkdir(exist_ok=True)

            # now run framewise alignment
            logging.info(f"Running run{run:02d}")
            # make stage
            stage_framewise_align = Stage(
                framewise_align,
                stage_name="framewise_align_func",
                hash_output=(hash_outputs / f"run{run:02d}").path,
            )
            stage_framewise_align.run(
                me_epi_ref_path, me_epi_mag_data[idx][0].path, (PathMan(f"run{run:02d}") / f"run{run:02d}").path
            )
    # do the same for fmaps
    fmap_aligned_output = output_dir / "framewise_align" / "fmap"
    fmap_aligned_output.mkdir(exist_ok=True, parents=True)
    with working_directory(fmap_aligned_output.path):
        for idx, run in zip(range(n_runs), layout.get_runs(datatype="fmap")):
            PathMan(f"run{run:02d}").mkdir(exist_ok=True)

            # now run framewise alignment
            logging.info(f"Running run{run:02d}")
            stage_framewise_align = Stage(
                framewise_align,
                stage_name="framewise_align_fmap",
                hash_output=(hash_outputs / f"run{run:02d}").path,
            )
            stage_framewise_align.run(
                pepolar_fmap_ref_path,
                pepolar_fmap_data[idx][func_pe].path,
                (PathMan(f"run{run:02d}") / f"run{run:02d}").path,
            )

    # now for each run, compute field maps
    medic_output = output_dir / "fieldmaps" / "medic"
    medic_output.mkdir(exist_ok=True, parents=True)
    with working_directory(medic_output.path):
        for idx, run in zip(range(n_runs), layout.get_runs(datatype="func")):
            PathMan(f"run{run:02d}").mkdir(exist_ok=True)
            # (hash_outputs / f"run{run:02d}" / "medic.stage").unlink(missing_ok=True)

            # get metadata
            echo_times = [me_epi_phase_data[idx][n].get_metadata()["EchoTime"] * 1000 for n in range(n_echos)]
            total_readout_time = me_epi_phase_data[idx][0].get_metadata()["TotalReadoutTime"]
            phase_encoding_direction = me_epi_phase_data[idx][0].get_metadata()["PhaseEncodingDirection"]

            # now run framewise alignment
            logging.info(f"Running run{run:02d}")
            stage_medic = Stage(run_medic, stage_name="medic", hash_output=(hash_outputs / f"run{run:02d}").path)
            stage_medic.run(
                [echo.path for echo in me_epi_phase_data[idx]],
                [echo.path for echo in me_epi_mag_data[idx]],
                echo_times,
                total_readout_time,
                phase_encoding_direction,
                int(run),
                border_size=3,
                svd_filt=5,
                critical_freq=None,
            )

    # do the same for fmaps
    topup_output = output_dir / "fieldmaps" / "topup"
    topup_output.mkdir(exist_ok=True, parents=True)
    with working_directory(topup_output.path):
        for idx, run in zip(range(n_runs), layout.get_runs(datatype="fmap")):
            PathMan(f"run{run:02d}").mkdir(exist_ok=True)

            # get total readout time
            total_readout_time = pepolar_fmap_data[idx][0].get_metadata()["TotalReadoutTime"]

            # concatenate fmaps
            pa_fmap, ap_fmap = pepolar_fmap_data[idx]
            pa_fmap_img = pa_fmap.get_image()
            ap_fmap_img = ap_fmap.get_image()
            pa_data = pa_fmap_img.get_fdata()
            ap_data = ap_fmap_img.get_fdata()
            concat_data = np.concatenate((pa_data, ap_data), axis=-1)
            fmaps = (PathMan(f"run{run:02d}") / "fmaps.nii.gz").path
            if not PathMan(fmaps).exists():
                nib.Nifti1Image(concat_data, pa_fmap_img.affine).to_filename(fmaps)

            # write acqparams file
            acqparams = (PathMan(f"run{run:02d}") / "acqparams.txt").path
            with open(acqparams, "w") as f:
                ped_idx = PED_TABLE[pa_fmap.get_metadata()["PhaseEncodingDirection"]]
                for _ in range(pa_data.shape[-1]):
                    f.write(f"{ped_idx} {total_readout_time}\n")
                ped_idx = PED_TABLE[ap_fmap.get_metadata()["PhaseEncodingDirection"]]
                for _ in range(ap_data.shape[-1]):
                    f.write(f"{ped_idx} {total_readout_time}\n")

            # now run framewise alignment
            logging.info(f"Running run{run:02d}")
            stage_topup = Stage(run_topup, stage_name="topup", hash_output=(hash_outputs / f"run{run:02d}").path)
            stage_topup.run(
                fmaps,
                acqparams,
                PathMan(f"run{run:02d}").path,
                (PathMan(f"run{run:02d}") / "iout").path,
                (PathMan(f"run{run:02d}") / "fout").path,
                (PathMan(f"run{run:02d}") / "dfout").path,
            )

    # apply transforms to fmaps
    medic_aligned_output = output_dir / "fieldmaps" / "medic_aligned"
    medic_aligned_output.mkdir(exist_ok=True, parents=True)
    with working_directory(medic_aligned_output.path):
        for idx, run in zip(range(n_runs), layout.get_runs(datatype="func")):
            PathMan(f"run{run:02d}").mkdir(exist_ok=True)
            stage_apply_framewise_mats = Stage(
                apply_framewise_mats,
                stage_name="apply_framewise_mats_func",
                hash_output=(hash_outputs / f"run{run:02d}").path,
            )
            stage_apply_framewise_mats.run(
                me_epi_ref_path,
                (medic_output / f"run{run:02d}" / "fmap.nii.gz").path,
                (func_aligned_output / f"run{run:02d}" / f"run{run:02d}.mat").path,
                (PathMan(f"run{run:02d}") / f"fmap").path,
            )
            stage_apply_framewise_mats = Stage(
                apply_framewise_mats,
                stage_name="apply_framewise_mats_func_native",
                hash_output=(hash_outputs / f"run{run:02d}").path,
            )
            stage_apply_framewise_mats.run(
                me_epi_ref_path,
                (medic_output / f"run{run:02d}" / "fmap.nii.gz").path,
                (func_aligned_output / f"run{run:02d}" / f"run{run:02d}.mat").path,
                (PathMan(f"run{run:02d}") / f"fmap_native").path,
            )
    topup_aligned_output = output_dir / "fieldmaps" / "topup_aligned"
    topup_aligned_output.mkdir(exist_ok=True, parents=True)
    with working_directory(topup_aligned_output.path):
        for idx, run in zip(range(n_runs), layout.get_runs(datatype="func")):
            PathMan(f"run{run:02d}").mkdir(exist_ok=True)
            stage_apply_framewise_mats = Stage(
                apply_framewise_mats,
                stage_name="apply_framewise_mats_fmap",
                hash_output=(hash_outputs / f"run{run:02d}").path,
            )
            stage_apply_framewise_mats.run(
                pepolar_fmap_ref_path,
                (topup_output / f"run{run:02d}" / "fout.nii.gz").path,
                (fmap_aligned_output / f"run{run:02d}" / f"run{run:02d}.mat").path,
                (PathMan(f"run{run:02d}") / f"run{run:02d}").path,
            )

    # apply corrections to data
    corrected_outputs = output_dir / "corrected_data"
    corrected_outputs.mkdir(exist_ok=True, parents=True)
    with working_directory(corrected_outputs.path):
        for idx, run in zip(range(n_runs), layout.get_runs(datatype="func")):
            current_run = PathMan(f"run{run:02d}")
            current_run.mkdir(exist_ok=True)

            # load the second echo data
            second_echo_path = me_epi_mag_data[idx][1].path
            second_echo_img = nib.load(second_echo_path)

            # load the reference img
            ref_img = nib.load(me_epi_ref_path)

            # do medic correction first
            medic_out = current_run / "medic"
            medic_out.mkdir(exist_ok=True, parents=True)
            with working_directory(medic_out.path):
                # load the medic displacement maps
                displacement_maps = output_dir / "fieldmaps" / "medic" / f"run{run:02d}" / "dmap.nii.gz"
                dmaps_img = nib.load(displacement_maps)

                # for each frame apply the correction
                corrected_data = np.zeros((*second_echo_img.shape[:3], args.num_frames))
                for frame_idx in range(args.num_frames):
                    logging.info(f"Correcting Frame: {frame_idx}")
                    # get the frame to correct
                    frame_data = second_echo_img.dataobj[..., frame_idx]

                    # make into image
                    frame_img = nib.Nifti1Image(frame_data, second_echo_img.affine)

                    # get the dmap for this frame
                    dmap_data = dmaps_img.dataobj[..., frame_idx]

                    # make into image
                    dmap_img = nib.Nifti1Image(dmap_data, dmaps_img.affine)

                    # get displacement field
                    dfield_img = displacement_map_to_field(dmap_img)

                    # apply the correction
                    corrected_img = resample_image(ref_img, frame_img, dfield_img)

                    # store the corrected_frame
                    corrected_data[..., frame_idx] = corrected_img.get_fdata()

                # save the corrected data
                corrected_img = nib.Nifti1Image(corrected_data, second_echo_img.affine)
                corrected_img.to_filename("medic_corrected.nii.gz")

                # now apply framewise alignments
                stage_apply_framewise_mats = Stage(
                    apply_framewise_mats,
                    stage_name="apply_framewise_mats_func_corrected",
                    hash_output=(hash_outputs / f"run{run:02d}").path,
                )
                stage_apply_framewise_mats.run(
                    "medic_corrected.nii.gz",
                    "medic_corrected.nii.gz",
                    (func_aligned_output / f"run{run:02d}" / f"run{run:02d}.mat").path,
                    "medic_corrected_aligned.nii.gz",
                )

            # now correct with topup
            topup_out = current_run / "topup"
            topup_out.mkdir(exist_ok=True, parents=True)
            with working_directory(topup_out.path):
                # load the topup displacment field (use first field)
                displacement_field = nib.load(output_dir / "fieldmaps" / "topup" / f"run{run:02d}" / "dfout_01.nii.gz")

                # apply framewise alignments to distorted 2nd echo
                stage_apply_framewise_mats = Stage(
                    apply_framewise_mats,
                    stage_name="apply_framewise_mats_func_distorted",
                    hash_output=(hash_outputs / f"run{run:02d}").path,
                )
                stage_apply_framewise_mats.run(
                    second_echo_path,
                    second_echo_path,
                    (func_aligned_output / f"run{run:02d}" / f"run{run:02d}.mat").path,
                    "topup_aligned.nii.gz",
                )
                topup_aligned_img = nib.load("topup_aligned.nii.gz")

                # for each frame apply the correction
                corrected_data = np.zeros((*topup_aligned_img.shape[:3], args.num_frames))
                for frame_idx in range(args.num_frames):
                    logging.info(f"Correcting Frame: {frame_idx}")

                    # get the data
                    field_data = displacement_field.get_fdata()

                    # make into image
                    field_img = nib.Nifti1Image(field_data, ref_img.affine)

                    # convert to itk format
                    itk_field_img = convert_warp(field_img, "fsl", "itk")

                    # get the frame to correct
                    frame_data = topup_aligned_img.dataobj[..., frame_idx]

                    # make into image
                    frame_img = nib.Nifti1Image(frame_data, topup_aligned_img.affine)

                    # apply the correction
                    corrected_img = resample_image(ref_img, frame_img, itk_field_img)

                    # store the corrected_frame
                    corrected_data[..., frame_idx] = corrected_img.get_fdata()
                # save the corrected data
                corrected_img = nib.Nifti1Image(corrected_data, second_echo_img.affine)
                corrected_img.to_filename("topup_aligned_corrected.nii.gz")
