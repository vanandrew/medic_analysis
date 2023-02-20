from memori.logging import run_process


def framewise_align(ref_frame, img_file, out_prefix):
    # call mcflirt
    run_process(
        [
            "mcflirt",
            "-in",
            img_file,
            "-r",
            ref_frame,
            "-out",
            out_prefix,
            "-stats",
            "-mats",
            "-plots",
            "-report",
        ]
    )


def apply_framewise_mats(ref_frame, img_file, mats_path, out_prefix, mat_prefix="MAT_"):
    # call applyxfm4D
    run_process(
        [
            "applyxfm4D",
            img_file,
            ref_frame,
            out_prefix,
            mats_path,
            "--fourdigit",
            "--userprefix",
            mat_prefix,
        ]
    )
