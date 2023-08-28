import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from pathlib import Path
from warpkit.utilities import create_brain_mask
from concurrent.futures import ProcessPoolExecutor, as_completed
from . import DATA_DIR


AA_DATA_DIR = Path("/data/Daenerys/ASD_ADHD/NP1173/derivatives/me_pipeline2")


def compute_tSNR(run_dir, pipeline, subject, session, run):
    # get the number of the run_dir
    run_num = run_dir.name.split("bold")[-1]
    # get the tSNR image
    tsnr_path = [f for f in run_dir.glob("*_Swgt_norm_SNR.nii.gz")][0]
    tsnr_img = nib.load(tsnr_path)
    tsnr_data = tsnr_img.get_fdata().squeeze()
    # get the average image
    avg_path = [f for f in run_dir.glob("*_Swgt_norm_avg.nii.gz")][0]
    avg_img = nib.load(avg_path)
    avg_data = avg_img.get_fdata().squeeze()
    # get the brain mask
    brain_mask = create_brain_mask(avg_data)

    # get the standard deviation image
    std_path = [f for f in run_dir.glob("*_Swgt_norm_sd1.nii.gz")][0]
    std_img = nib.load(std_path)
    std_data = std_img.get_fdata().squeeze()

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
    std_masked_data = data_std[std_mask]
    mean_std_masked = np.mean(std_masked_data)
    mean_tsnr = np.mean(tsnr_data[brain_mask])
    mean_std = np.mean(std_data[brain_mask])
    return {
        "pipeline": pipeline,
        "subject": subject,
        "session": session,
        "run": run,
        "good_frames": good_frames,
        "percent_good_frames": percent_good_frames,
        "num_frames": tmask.size,
        "mean_tsnr": mean_tsnr,
        "mean_std": mean_std,
        "mean_tsnr_masked": mean_tsnr_masked,
        "mean_std_masked": mean_std_masked,
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
        "mean_tsnr": [],
        "mean_std": [],
        "mean_tsnr_masked": [],
        "mean_std_masked": [],
    }
    futures = []
    with ProcessPoolExecutor(max_workers=10) as executor:
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
                        executor.submit(compute_tSNR, run_dir, pipeline, subject_dir.name, session_name, run_dir.name)
                    )
                    # futures.append(
                    #     compute_tSNR(run_dir, pipeline, subject_dir.name, session_name, run_dir.name)
                    # )
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
            datalist["mean_tsnr"].append(result["mean_tsnr"])
            datalist["mean_std"].append(result["mean_std"])
            datalist["mean_tsnr_masked"].append(result["mean_tsnr_masked"])
            datalist["mean_std_masked"].append(result["mean_std_masked"])

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
    df["difference_tsnr"] = df["mean_tsnr_medic"] - df["mean_tsnr_topup"]
    df["difference_std"] = df["mean_std_medic"] - df["mean_std_topup"]
    df["difference_tsnr_masked"] = df["mean_tsnr_masked_medic"] - df["mean_tsnr_masked_topup"]
    df["difference_std_masked"] = df["mean_std_masked_medic"] - df["mean_std_masked_topup"]
    df.to_csv(str(DATA_DIR / "tsnr.csv"), index=False)
    # temporary fix for bad runs
    print(ttest_rel(df.mean_tsnr_medic, df.mean_tsnr_topup))
    print(ttest_rel(df.mean_std_medic, df.mean_std_topup))
    print(ttest_rel(df.mean_tsnr_masked_medic, df.mean_tsnr_masked_topup))
    print(ttest_rel(df.mean_std_masked_medic, df.mean_std_masked_topup))
    from IPython import embed

    embed()
