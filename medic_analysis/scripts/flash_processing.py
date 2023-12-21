"""Prepare data for FLASH analysis."""
import shutil

import nibabel as nib
from bids import BIDSLayout
from memori.helpers import working_directory
from memori.logging import run_process
from memori.pathman import PathManager as PathMan
from memori.stage import Stage

from medic_analysis.common.align import framewise_align
from medic_analysis.common.bias_field import N4BiasCorrection
from medic_analysis.common.distortion import run_medic, run_romeo, run_topup

from . import PED_TABLE, parser

# Define the path to the BIDS dataset
BIDS_DATA_DIR = "/home/usr/vana/GMT2/Andrew/FLASHSUSTEST"


def main():
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

    # get the flash scan (both runs)
    flash_mag1 = layout.get(
        datatype="fmap",
        suffix="magnitude1",
        extension="nii.gz",
    )
    flash_mag2 = layout.get(
        datatype="fmap",
        suffix="magnitude2",
        extension="nii.gz",
    )
    flash_phase1 = layout.get(
        datatype="fmap",
        suffix="phase1",
        extension="nii.gz",
    )
    flash_phase2 = layout.get(
        datatype="fmap",
        suffix="phase2",
        extension="nii.gz",
    )

    # get the SE fmap (only first run)
    se_fmaps = [
        layout.get(datatype="fmap", suffix="epi", extension="nii.gz", direction="PA", run="01")[0],
        layout.get(datatype="fmap", suffix="epi", extension="nii.gz", direction="AP", run="01")[0],
    ]

    # get the ME-EPI data (only first run)
    me_epi_mag = layout.get(datatype="func", suffix="bold", task="rest", extension="nii.gz", part="mag", run="02")
    me_epi_phase = layout.get(datatype="func", suffix="bold", task="rest", extension="nii.gz", part="phase", run="02")

    # make directory for hash outputs
    hash_outputs = output_dir / "hash_outputs"
    hash_outputs.mkdir(exist_ok=True)

    # process flash scans
    flash1_dir = output_dir / "flash1"
    flash1_dir.mkdir(exist_ok=True)
    flash2_dir = output_dir / "flash2"
    flash2_dir.mkdir(exist_ok=True)
    out_dirs = [flash1_dir, flash2_dir]
    runs = ["01", "02"]
    for idx, run in enumerate(runs):
        out_dir = out_dirs[idx]
        with working_directory(out_dir.path):
            flash_hashout = hash_outputs / out_dir.name
            flash_hashout.mkdir(exist_ok=True)

            # first flash scan needs bias correction
            if idx == 0:
                stage_n4_bias1 = Stage(
                    N4BiasCorrection,
                    stage_name="n4_bias1",
                    hash_output=(flash_hashout / "n4_bias1").path,
                )
                stage_n4_bias1.run(flash_mag1[0].path, "mag1_biascorrected.nii.gz")
                stage_n4_bias2 = Stage(
                    N4BiasCorrection,
                    stage_name="n4_bias2",
                    hash_output=(flash_hashout / "n4_bias2").path,
                )
                stage_n4_bias2.run(flash_mag2[0].path, "mag2_biascorrected.nii.gz")
            else:
                shutil.copyfile(flash_mag1[idx].path, "mag1_biascorrected.nii.gz")
                shutil.copyfile(flash_mag2[idx].path, "mag2_biascorrected.nii.gz")

            # clip the images to 500
            if not PathMan("mag1_biascorrected_clipped.nii.gz").exists():
                nib.Nifti1Image(
                    nib.load("mag1_biascorrected.nii.gz").get_fdata().clip(0, 500),
                    nib.load("mag1_biascorrected.nii.gz").affine,
                ).to_filename("mag1_biascorrected_clipped.nii.gz")
            if not PathMan("mag2_biascorrected_clipped.nii.gz").exists():
                nib.Nifti1Image(
                    nib.load("mag2_biascorrected.nii.gz").get_fdata().clip(0, 500),
                    nib.load("mag2_biascorrected.nii.gz").affine,
                ).to_filename("mag2_biascorrected_clipped.nii.gz")

            # get brain mask
            run_process(
                [
                    "bet",
                    "mag1_biascorrected_clipped.nii.gz",
                    "brain.nii.gz",
                    "-f",
                    "0.4",
                    "-m",
                    "-v",
                ]
            )

            # now concatenate the mags and phases
            if not PathMan("combined_mag.nii.gz").exists() and not PathMan("combined_phase.nii.gz").exists():
                mag1 = nib.load("mag1_biascorrected_clipped.nii.gz")
                mag2 = nib.load("mag2_biascorrected_clipped.nii.gz")
                phase1 = nib.load(flash_phase1[0].path)
                phase2 = nib.load(flash_phase2[0].path)
                combined_mag = nib.concat_images([mag1, mag2])
                combined_phase = nib.concat_images([phase1, phase2])
                combined_mag.to_filename("combined_mag.nii.gz")
                combined_phase.to_filename("combined_phase.nii.gz")

            # get the TE values
            echo_times = [
                flash_mag1[idx].get_metadata()["EchoTime"] * 1000,
                flash_mag2[idx].get_metadata()["EchoTime"] * 1000,
            ]

            # now call romeo on data
            romeo_stage = Stage(run_romeo, stage_name="romeo", hash_output=(flash_hashout / "romeo").path)
            romeo_stage.run(
                "combined_phase.nii.gz",
                "combined_mag.nii.gz",
                echo_times,
                "romeo_output",
            )

    # now process the SE field map
    se_fmap_dir = output_dir / "topup"
    se_fmap_dir.mkdir(exist_ok=True)
    with working_directory(se_fmap_dir.path):
        se_fmap_hashout = hash_outputs / se_fmap_dir.name
        se_fmap_hashout.mkdir(exist_ok=True)

        # grab the readout time and phase encoding directions
        total_readout_time = se_fmaps[0].get_metadata()["TotalReadoutTime"]
        phase_encoding_directions = [
            se_fmaps[0].get_metadata()["PhaseEncodingDirection"],
            se_fmaps[1].get_metadata()["PhaseEncodingDirection"],
        ]

        # write acqparams file
        with open("acqparams.txt", "w") as f:
            ped_idx = PED_TABLE[phase_encoding_directions[0]]
            for _ in range(se_fmaps[0].get_image().shape[-1]):
                f.write(f"{ped_idx} {total_readout_time}\n")
            ped_idx = PED_TABLE[phase_encoding_directions[1]]
            for _ in range(se_fmaps[1].get_image().shape[-1]):
                f.write(f"{ped_idx} {total_readout_time}\n")
        acqparams = "acqparams.txt"

        # now concatenate the data
        if not PathMan("imain.nii.gz").exists():
            imain = nib.concat_images([se_fmaps[0].get_image(), se_fmaps[1].get_image()], axis=3)
            imain.to_filename("imain.nii.gz")
        imain = "imain.nii.gz"

        # call topup
        topup_stage = Stage(
            run_topup,
            stage_name="topup",
            hash_output=(se_fmap_hashout / "topup").path,
        )
        topup_stage.run(imain, acqparams, "./", "iout", "fout", "dfout")

    # run medic
    medic_dir = output_dir / "medic"
    medic_dir.mkdir(exist_ok=True)
    with working_directory(medic_dir.path):
        medic_hashout = hash_outputs / medic_dir.name
        medic_hashout.mkdir(exist_ok=True)

        # get echo times, total readout time, and phase encoding direction
        echo_times = [p.get_metadata()["EchoTime"] * 1000 for p in me_epi_phase]
        total_readout_time = me_epi_phase[0].get_metadata()["TotalReadoutTime"]
        phase_encoding_direction = me_epi_phase[0].get_metadata()["PhaseEncodingDirection"]

        # get first frame
        first_echo = me_epi_mag[0].get_image()
        first_frame = nib.Nifti1Image(first_echo.dataobj[..., 0], first_echo.affine, first_echo.header)
        first_frame.to_filename("ref_frame.nii.gz")

        # run mcflirt
        mcflirt_stage = Stage(
            framewise_align,
            stage_name="mcflirt",
            hash_output=(medic_hashout / "mcflirt").path,
        )
        mcflirt_stage.run(
            "ref_frame.nii.gz",
            me_epi_mag[0].path,
            "me_framewise_align",
        )

        # run medic
        medic_stage = Stage(
            run_medic,
            stage_name="medic",
            hash_output=(medic_hashout / "medic").path,
        )
        PathMan("run01").mkdir(exist_ok=True)
        medic_stage.run(
            [p.path for p in me_epi_phase],
            [p.path for p in me_epi_mag],
            echo_times,
            total_readout_time,
            phase_encoding_direction,
            1,
            border_size=3,
            svd_filt=5,
        )
        if PathMan("run01").exists():
            for f in PathMan("run01").glob("*"):
                shutil.move(f, PathMan("./") / f.name)
            shutil.rmtree("run01")

    # resample fmaps to high res flash space
    run_process(
        [
            "antsApplyTransforms",
            "-d",
            "4",
            "-o",
            str(PathMan(args.output_dir) / "topup" / "fout_hres.nii.gz"),
            "-r",
            str(PathMan(args.output_dir) / "flash1" / "romeo_output" / "B0.nii"),
            "-i",
            str(PathMan(args.output_dir) / "topup" / "fout.nii.gz"),
            "-n",
            "LanczosWindowedSinc",
            "-v",
            "1",
        ]
    )
    run_process(
        [
            "antsApplyTransforms",
            "-d",
            "4",
            "-o",
            str(PathMan(args.output_dir) / "medic" / "fmap_hres.nii.gz"),
            "-r",
            str(PathMan(args.output_dir) / "flash1" / "romeo_output" / "B0.nii"),
            "-i",
            str(PathMan(args.output_dir) / "medic" / "fmap.nii.gz"),
            "-n",
            "LanczosWindowedSinc",
            "-v",
            "1",
        ]
    )
