"""Functions to perform bias field correction."""
from memori.logging import run_process


def N4BiasCorrection(img, out_file):
    run_process(
        [
            "N4BiasFieldCorrection",
            "3",
            "-i",
            f"{img}",
            "-o",
            f"{out_file}",
            "-b",
            "[100,3,1x1x1,3]",
            "-v",
        ],
    )
