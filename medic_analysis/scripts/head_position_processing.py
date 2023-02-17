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
    medic_call,
)
from warpkit.utilities import displacement_map_to_field
from . import (
    parser,
    PED_TABLE,
    POLARITY_IDX,
)

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

# Define constants for run numbers and indices
STATIC_HEAD_POSITION_RUN_NUMBER = [1, 2, 3, 4, 5, 6, 7, 15]
STATIC_HEAD_POSITION_RUN_IDX = [0, 1, 2, 3, 4, 5, 6, 14]
TRANSIENT_HEAD_POSITION_RUN_NUMBER = [8, 9, 10, 11, 12, 13, 14]
TRANSIENT_HEAD_POSITION_RUN_IDX = [7, 8, 9, 10, 11, 12, 13]

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

    # get total number of rest runs in data
    n_runs = len(layout.get_runs(datatype="func"))
    assert n_runs == len(layout.get_runs(datatype="fmap")), "fmaps should have equal number of runs as func"

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

            # get metadata
            echo_times = [me_epi_phase_data[idx][n].get_metadata()["EchoTime"] * 1000 for n in range(n_echos)]
            total_readout_time = me_epi_phase_data[idx][0].get_metadata()["TotalReadoutTime"]
            phase_encoding_direction = me_epi_phase_data[idx][0].get_metadata()["PhaseEncodingDirection"]

            # now run framewise alignment
            logging.info(f"Running run{run:02d}")
            stage_medic = Stage(medic_call, stage_name="medic", hash_output=(hash_outputs / f"run{run:02d}").path)
            stage_medic.run(
                [echo.path for echo in me_epi_phase_data[idx]],
                [echo.path for echo in me_epi_mag_data[idx]],
                echo_times,
                total_readout_time,
                phase_encoding_direction,
                int(run),
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
            with open("acqparams.txt", "w") as f:
                ped_idx = PED_TABLE[pa_fmap.get_metadata()["PhaseEncodingDirection"]]
                for _ in range(pa_data.shape[-1]):
                    f.write(f"{ped_idx} {total_readout_time}\n")
                ped_idx = PED_TABLE[ap_fmap.get_metadata()["PhaseEncodingDirection"]]
                for _ in range(ap_data.shape[-1]):
                    f.write(f"{ped_idx} {total_readout_time}\n")
            acqparams = (PathMan(f"run{run:02d}") / "acqparams.txt").path

            # now run framewise alignment
            logging.info(f"Running run{run:02d}")
            stage_topup = Stage(run_topup, stage_name="topup", hash_output=(hash_outputs / f"run{run:02d}").path)
            stage_topup.run(
                fmaps,
                acqparams,
                PathMan(f"run{run:02d}").path,
                (PathMan(f"run{run:02d}") / "fout").path,
                (PathMan(f"run{run:02d}") / "iout").path,
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
                (PathMan(f"run{run:02d}") / f"run{run:02d}").path,
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
    corrected_medic_output = output_dir / "corrected_data"
    corrected_medic_output.mkdir(exist_ok=True, parents=True)
    with working_directory(corrected_medic_output.path):
        for idx, run in zip(range(n_runs), layout.get_runs(datatype="func")):
            PathMan(f"run{run:02d}").mkdir(exist_ok=True)

            # we only want to correct the first echo
            # so grab the path for it
            first_echo_path = me_epi_mag_data[idx][0].path
