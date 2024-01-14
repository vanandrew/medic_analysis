"""Computes the tSNR analysis for each pipeline

This module is similar to the `alignment_metics.py` but does the tSNR analysis
for each pipeline.

This module expects derivative outputs from the dosenbach lab preprocessing pipeline:
https://github.com/DosenbachGreene/processing_pipeline
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from warpkit.utilities import create_brain_mask

from . import DATA_DIR

AA_DATA_DIR = Path("/data/Daenerys/ASD_ADHD/NP1173/derivatives/me_pipeline2")


def compute_tSNR(run_dir, pipeline, subject, session, run):
    # get the average image
    avg_path = [f for f in run_dir.glob("*_Swgt_norm_avg.nii.gz")][0]
    avg_img = nib.load(avg_path)
    avg_data = avg_img.get_fdata().squeeze()
    # get the brain mask
    brain_mask = create_brain_mask(avg_data)

    # get the tmask
    tmask_path = [f for f in run_dir.glob("*_tmask.txt")][0]
    with open(tmask_path, "r") as f:
        tmask = np.array([np.round(float(line.strip())).astype(bool) for line in f.readlines()])
    # compute percent good frames
    good_frames = tmask.sum()
    percent_good_frames = good_frames / tmask.size
    # get the time series
    time_series_path = [f for f in run_dir.glob("*_Swgt_norm.nii")][0]
    time_series_img = nib.load(time_series_path)
    # only grab the frames in tmask
    time_series_data = time_series_img.get_fdata()[..., tmask]
    # now get data only in brain mask
    time_series_data = time_series_data[brain_mask, :]
    # get the mean of the data
    data_mean = np.mean(time_series_data, axis=-1)
    # get the std dev of the data
    data_std = np.std(time_series_data, axis=-1)
    # mask where std dev is 0
    std_mask = data_std != 0
    # get masked tSNR and std
    tsnr_masked_data = data_mean[std_mask] / data_std[std_mask]
    mean_tsnr_masked = np.mean(tsnr_masked_data)
    return {
        "pipeline": pipeline,
        "subject": subject,
        "session": session,
        "run": run,
        "good_frames": good_frames,
        "percent_good_frames": percent_good_frames,
        "num_frames": tmask.size,
        "mean_tsnr_masked": mean_tsnr_masked,
    }


def main():
    datalist = {
        "pipeline": [],
        "subject": [],
        "session": [],
        "run": [],
        "good_frames": [],
        "percent_good_frames": [],
        "num_frames": [],
        "mean_tsnr_masked": [],
    }
    futures = []
    with ProcessPoolExecutor(max_workers=100) as executor:
        # loop over subjects in AA_DATA_DIR
        for subject_dir in sorted(AA_DATA_DIR.glob("sub-*")):
            # print(subject_dir.name)
            for session_dir in sorted(subject_dir.glob("ses-*")):
                # print(session_dir.name)
                pipeline = "MEDIC"
                session_name = session_dir.name
                if "TOPUP" in session_dir.name:
                    pipeline = "TOPUP"
                    session_name = session_dir.name.split("wTOPUP")[0]
                # for each run get the tSNR image
                for run_dir in sorted(session_dir.glob("bold*")):
                    # print(run_dir.name)
                    print(f"Submitting Job: {subject_dir.name}, {session_name}, {run_dir.name}")
                    futures.append(
                        executor.submit(
                            compute_tSNR,
                            run_dir,
                            pipeline,
                            subject_dir.name,
                            session_name,
                            run_dir.name,
                        )
                    )
                    print(f"Submitted Job: {subject_dir.name}, {session_name}, {run_dir.name}")

        for future in as_completed(futures):
            # for future in futures:
            print(f"Getting Result: {future}")
            result = future.result()
            # result = future
            print(f"Finished Job: {result}")
            datalist["pipeline"].append(result["pipeline"])
            datalist["subject"].append(result["subject"])
            datalist["session"].append(result["session"])
            datalist["run"].append(result["run"])
            datalist["good_frames"].append(result["good_frames"])
            datalist["percent_good_frames"].append(result["percent_good_frames"])
            datalist["num_frames"].append(result["num_frames"])
            datalist["mean_tsnr_masked"].append(result["mean_tsnr_masked"])

    # get dataframe
    df = pd.DataFrame(datalist)
    # get MEDIC pipeline and TOPUP pipeline separately
    medic_df = df[df["pipeline"] == "MEDIC"]
    topup_df = df[df["pipeline"] == "TOPUP"]
    # drop pipeline column
    medic_df = medic_df.drop(columns=["pipeline"])
    topup_df = topup_df.drop(columns=["pipeline"])
    # merge the two dataframes on subject, session, and run
    df = pd.merge(medic_df, topup_df, on=["subject", "session", "run"], suffixes=("_medic", "_topup"))
    df["difference_tsnr_masked"] = df["mean_tsnr_masked_medic"] - df["mean_tsnr_masked_topup"]
    df.to_csv(str(DATA_DIR / "tsnr.csv"), index=False)
    # temporary fix for bad runs
    print(ttest_rel(df.mean_tsnr_masked_medic, df.mean_tsnr_masked_topup))
    # from IPython import embed
    # embed()
