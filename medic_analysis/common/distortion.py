import os
import numpy as np
import nibabel as nib
from memori.logging import run_process
from memori.pathman import PathManager as PathMan
from warpkit.distortion import medic
from typing import cast


if os.environ.get("ROMEO_PATH") is None:
    ROMEO_PATH = "/home/usr/vana/GMT2/Andrew/romeo/compiled/bin/romeo"
else:
    ROMEO_PATH = cast(str, os.environ.get("ROMEO_PATH"))


def run_medic(
    phases,
    mag,
    echo_times,
    total_readout_time,
    phase_encoding_direction,
    run,
    num_frames=100,
    **kwargs,
):
    phase_imgs = [nib.load(i) for i in phases]
    mag_imgs = [nib.load(i) for i in mag]
    fmap_native, dmap, fmap = medic(
        phase_imgs,
        mag_imgs,
        echo_times,
        total_readout_time,
        phase_encoding_direction,
        frames=[i for i in range(num_frames)],
        **kwargs,
    )
    # save the data
    fmap_native.to_filename(PathMan(f"run{run:02d}") / "fmap_native.nii.gz")
    dmap.to_filename(PathMan(f"run{run:02d}") / "dmap.nii.gz")
    fmap.to_filename(PathMan(f"run{run:02d}") / "fmap.nii.gz")


def run_topup(fmaps, acqparams, out_path, iout_path, fout_path, dfout_path):
    run_process(
        [
            "topup",
            f"--imain={fmaps}",
            f"--datain={acqparams}",
            "--config=b02b0.cnf",
            f"--out={out_path}",
            f"--iout={iout_path}",
            f"--fout={fout_path}",
            f"--dfout={dfout_path}",
            "-v",
        ]
    )


def applywarp(ref, in_file, out_file, warp, interp="sinc"):
    run_process(
        [
            "applywarp",
            f"--ref={ref}",
            f"--in={in_file}",
            f"--out={out_file}",
            f"--warp={warp}",
            f"--interp={interp}",
            "-v",
        ]
    )


def run_romeo(phase, mag, echo_times, out):
    run_process(
        [
            ROMEO_PATH,
            "-p",
            phase,
            "-m",
            mag,
            "-t",
            "[" + ",".join([str(e) for e in echo_times]) + "]",
            "-i",
            "-g",
            "-B",
            "--phase-offset-smoothing-sigma-mm",
            "[7,7,7]",
            "--write-phase-offsets",
            "-o",
            out,
            "--verbose",
        ]
    )
