import numpy as np
import scipy.sparse as ss
from pathlib import Path
import nibabel as nib
from warpkit.utilities import create_brain_mask
import pandas as pd
from scipy.stats import ttest_rel
from skimage.morphology import ball
from skimage.filters import gaussian
from concurrent.futures import ProcessPoolExecutor, as_completed
from medic_analysis.common.align import roc_metrics
from . import DATA_DIR


ASD_ADHD_DATA = Path("/home/usr/vana/Daenerys/ASD_ADHD/NP1173/derivatives/me_pipeline2")


def correlation(data1, data2):
    corr = np.corrcoef(data1, data2)[0, 1]
    return corr if not np.isnan(corr) else 0


def correlation2(data1, data2):
    return correlation(data1, data2) ** 2


def rolling_window(data1, data2, window, mask, func):
    # ensure the data sizes and dimensions are the same
    assert data1.shape == data2.shape, "Data1 and data2 must have the same shape."
    assert data1.ndim == data2.ndim, "Data1 and data2 must have the same number of dimensions."

    # window dimensions must be odd
    assert np.all(np.mod(window.shape, 2) == 1), "Window dimensions must be odd."

    # compute the indices of the window where it is 1
    window_indices = np.argwhere(window == 1)

    # find the center of the window
    window_center = np.array([window.shape[0] // 2, window.shape[1] // 2, window.shape[2] // 2])

    # offset the window indices by the window center
    window_indices = window_indices - window_center

    new_data = np.zeros_like(data1)

    # start the window at the beginning of the data
    for i in range(data1.shape[0]):
        for j in range(data1.shape[1]):
            for k in range(data1.shape[2]):
                center = np.array([i, j, k])
                indices = center - window_indices
                # check if the indices are within the bounds of the data
                # remove indices that are out of bounds
                indices = indices[np.all(indices >= 0, axis=1)]
                indices = indices[np.all(indices < data1.shape, axis=1)]
                # now get the mask values for the indices
                submask = mask[indices[:, 0], indices[:, 1], indices[:, 2]]
                # if mask doesn't have at least two Trues, then skip
                if np.sum(submask) < 2:
                    continue
                # now grab the data
                subdata1 = data1[indices[:, 0], indices[:, 1], indices[:, 2]]
                subdata2 = data2[indices[:, 0], indices[:, 1], indices[:, 2]]
                # and apply mask
                subdata1 = subdata1[submask]
                subdata2 = subdata2[submask]
                # compute function on window
                new_data[i, j, k] = func(subdata1, subdata2)
    # return the new data
    return new_data


def nmi(x, y):
    """
    Compute normalized mutual information I(x,y)/sqrt(H(x)*H(y)) of two
    discrete variables x and y.

    Parameters:
    - x, y: two integer vectors of the same length
    Returns:
    - z: normalized mutual information z=I(x,y)/sqrt(H(x)*H(y))
    """
    x, _ = np.histogram(x, bins=256)
    y, _ = np.histogram(y, bins=256)
    assert x.size == y.size, "Input vectors x and y must have the same length."
    if x.size < 3:
        return np.nan
    n = x.size
    x = x.ravel()
    y = y.ravel()
    offset = min(np.min(x), np.min(y))
    x = x - offset
    y = y - offset
    k = max(np.max(x), np.max(y)) + 1
    idx = np.arange(n)
    Mx = ss.coo_matrix((np.ones(n), (idx, x)), shape=(n, k)).tocsc()
    My = ss.coo_matrix((np.ones(n), (idx, y)), shape=(n, k)).tocsc()
    Pxy = (Mx.transpose().dot(My) / n).data
    Hxy = -np.sum(Pxy * np.log2(Pxy))
    mean_Mx = np.mean(Mx, axis=0).A1
    mean_My = np.mean(My, axis=0).A1
    nonzero_Mx = np.nonzero(mean_Mx)
    nonzero_My = np.nonzero(mean_My)
    Px = mean_Mx[nonzero_Mx]
    Py = mean_My[nonzero_My]
    Hx = -np.dot(Px, np.log2(Px))
    Hy = -np.dot(Py, np.log2(Py))
    MI = Hx + Hy - Hxy
    z = np.sqrt((MI / Hx) * (MI / Hy))
    return max(0, z)


def compute_metrics(bold_dir, t1_path, t2_path, wmparc_path, pipeline, subject, session, run):
    # load the wmparc file
    wmparc_img = nib.load(wmparc_path)
    wmparc_data = wmparc_img.get_fdata().squeeze()
    wmparc_brain = wmparc_data != 0

    # get the average image
    avg_path = [f for f in bold_dir.glob("*_Swgt_norm.nii")][0]
    # avg_path = [f for f in bold_dir.glob("*_Swgt_norm_avg.nii.gz")][0]
    avg_img = nib.load(avg_path)
    avg_data = avg_img.dataobj[..., 0]
    # avg_data = avg_img.get_fdata().squeeze()
    # get the brain mask
    brain_mask = create_brain_mask(avg_data)

    # get the t1 and t2 images
    t1_img = nib.load(t1_path)
    t1_data = t1_img.get_fdata().squeeze()
    t2_img = nib.load(t2_path)
    t2_data = t2_img.get_fdata().squeeze()

    # take correlation between t1/t2 and bold
    corr_t1 = correlation2(t1_data[brain_mask], avg_data[brain_mask])
    corr_t2 = correlation(t2_data[brain_mask], avg_data[brain_mask])

    # compute gradients of data
    grad_t1_data = np.sqrt(np.sum(np.stack(np.gradient(t1_data), axis=-1) ** 2, axis=-1))
    grad_t2_data = np.sqrt(np.sum(np.stack(np.gradient(t2_data), axis=-1) ** 2, axis=-1))
    grad_avg_data = np.sqrt(np.sum(np.stack(np.gradient(avg_data), axis=-1) ** 2, axis=-1))

    # take correlation between gradients
    grad_corr_t1 = correlation(grad_t1_data[brain_mask], grad_avg_data[brain_mask])
    grad_corr_t2 = correlation(grad_t2_data[brain_mask], grad_avg_data[brain_mask])

    # get normalized mutual information
    nmi_t1 = nmi(t1_data[brain_mask], avg_data[brain_mask])
    nmi_t2 = nmi(t2_data[brain_mask], avg_data[brain_mask])

    # create spotlight element
    spotlight = ball(3)
    local_corr_t1 = rolling_window(t1_data, avg_data, spotlight, brain_mask, correlation2) * brain_mask
    local_corr_t2 = rolling_window(t2_data, avg_data, spotlight, brain_mask, correlation2) * brain_mask
    local_corr_mean_t1 = local_corr_t1.mean()
    local_corr_mean_t2 = local_corr_t2.mean()

    # compute ROC metrics
    roc_gw, roc_ie, roc_vw, roc_cb_ie = roc_metrics(avg_data, wmparc_data)

    # return the results
    return {
        "pipeline": pipeline,
        "subject": subject,
        "session": session,
        "run": run,
        "corr_t1": corr_t1,
        "corr_t2": corr_t2,
        "grad_corr_t1": grad_corr_t1,
        "grad_corr_t2": grad_corr_t2,
        "nmi_t1": nmi_t1,
        "nmi_t2": nmi_t2,
        "local_corr_t1": local_corr_t1,
        "local_corr_t2": local_corr_t2,
        "local_corr_mean_t1": local_corr_mean_t1,
        "local_corr_mean_t2": local_corr_mean_t2,
        "roc_gw": roc_gw,
        "roc_ie": roc_ie,
        "roc_vw": roc_vw,
        "roc_cb_ie": roc_cb_ie,
    }


def main():
    # make list to store the results
    datalist = {
        "pipeline": [],
        "subject": [],
        "session": [],
        "run": [],
        "corr_t1": [],
        "corr_t2": [],
        "grad_corr_t1": [],
        "grad_corr_t2": [],
        "nmi_t1": [],
        "nmi_t2": [],
        "local_corr_mean_t1": [],
        "local_corr_mean_t2": [],
        "roc_gw": [],
        "roc_ie": [],
        "roc_vw": [],
        "roc_cb_ie": [],
    }

    # create dicts for local correlation
    local_corr_t1_medic = {}
    local_corr_t2_medic = {}
    local_corr_t1_topup = {}
    local_corr_t2_topup = {}

    # create a ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=10) as executor:
        # create futures list
        futures = []

        # loop over subjects
        for subject_dir in sorted(ASD_ADHD_DATA.glob("sub-*")):
            # get the subject's wmparc file
            wmparc_path = next((subject_dir / "T1" / "atlas").glob("*_wmparc_on_MNI152_T1_2mm.nii.gz"))

            # get the T1w image
            t1_path = next((subject_dir / "T1" / "atlas").glob("*_T1w*debias*_on_MNI152_T1_2mm.nii.gz"))

            # get the T2w image
            t2_path = next((subject_dir / "T1" / "atlas").glob("*_T2w*debias*_on_MNI152_T1_2mm.nii.gz"))

            # loop over the sessions for the subject
            for session_dir in sorted(subject_dir.glob("ses-*")):
                # is topup
                isTOPUP = False
                session_name = session_dir.name
                if "wTOPUP" in session_dir.name:
                    isTOPUP = True
                    session_name = session_dir.name.split("w")[0]
                if "GRE" in session_dir.name:
                    continue
                # loop over each bold run
                for bold_dir in sorted(session_dir.glob("bold?")):
                    # get the number of the run_dir
                    run_num = bold_dir.name.split("bold")[-1]
                    label = "TOPUP" if isTOPUP else "MEDIC"
                    print(f"Submitting job for {label}, {subject_dir.name}, {session_name}, {run_num}")
                    # futures.append(
                    #     compute_metrics(
                    #         bold_dir, t1_path, t2_path, wmparc_path, label, subject_dir.name, session_name, run_num
                    #     )
                    # )
                    # break
                    futures.append(
                        executor.submit(
                            compute_metrics,
                            bold_dir,
                            t1_path,
                            t2_path,
                            wmparc_path,
                            label,
                            subject_dir.name,
                            session_name,
                            run_num,
                        )
                    )
            # break

        # loop over futures
        # for future in futures:
        for future in as_completed(futures):
            # get the results
            results = future.result()

            # print current pipeline subject, session, and run
            print(results["pipeline"])
            print(results["subject"])
            print(results["session"])
            print(results["run"])
            # print(results)

            # add stats to datalist
            datalist["pipeline"].append(results["pipeline"])
            datalist["subject"].append(results["subject"])
            datalist["session"].append(results["session"])
            datalist["run"].append(results["run"])
            datalist["corr_t1"].append(results["corr_t1"])
            datalist["corr_t2"].append(results["corr_t2"])
            datalist["grad_corr_t1"].append(results["grad_corr_t1"])
            datalist["grad_corr_t2"].append(results["grad_corr_t2"])
            datalist["nmi_t1"].append(results["nmi_t1"])
            datalist["nmi_t2"].append(results["nmi_t2"])
            datalist["local_corr_mean_t1"].append(results["local_corr_mean_t1"])
            datalist["local_corr_mean_t2"].append(results["local_corr_mean_t2"])
            datalist["roc_gw"].append(results["roc_gw"])
            datalist["roc_ie"].append(results["roc_ie"])
            datalist["roc_vw"].append(results["roc_vw"])
            datalist["roc_cb_ie"].append(results["roc_cb_ie"])

            # add local correlation to lists
            key = f"{results['subject']}_{results['session']}_{results['run']}"
            if results["pipeline"] == "MEDIC":
                local_corr_t1_medic[key] = results["local_corr_t1"]
                local_corr_t2_medic[key] = results["local_corr_t2"]
            else:
                local_corr_t1_topup[key] = results["local_corr_t1"]
                local_corr_t2_topup[key] = results["local_corr_t2"]

    # sort local correlations by key
    local_corr_t1_medic = [v for _, v in sorted(local_corr_t1_medic.items(), key=lambda item: item[0])]
    local_corr_t2_medic = [v for _, v in sorted(local_corr_t2_medic.items(), key=lambda item: item[0])]
    local_corr_t1_topup = [v for _, v in sorted(local_corr_t1_topup.items(), key=lambda item: item[0])]
    local_corr_t2_topup = [v for _, v in sorted(local_corr_t2_topup.items(), key=lambda item: item[0])]

    # convert local correlation lists to arrays
    local_corr_t1_medic = np.stack(local_corr_t1_medic, axis=-1)
    local_corr_t2_medic = np.stack(local_corr_t2_medic, axis=-1)
    local_corr_t1_topup = np.stack(local_corr_t1_topup, axis=-1)
    local_corr_t2_topup = np.stack(local_corr_t2_topup, axis=-1)

    # compute ttests for local correlation
    local_corr_t1 = ttest_rel(local_corr_t1_medic, local_corr_t1_topup, axis=-1)
    local_corr_t2 = ttest_rel(local_corr_t2_medic, local_corr_t2_topup, axis=-1)
    local_corr_t1_mask = local_corr_t1.pvalue < 0.05
    local_corr_t2_mask = local_corr_t2.pvalue < 0.05
    local_corr_t1_tstat = local_corr_t1.statistic * local_corr_t1_mask
    local_corr_t2_tstat = local_corr_t2.statistic * local_corr_t2_mask
    # get rid of nans
    local_corr_t1_tstat = np.nan_to_num(local_corr_t1_tstat)
    local_corr_t2_tstat = np.nan_to_num(local_corr_t2_tstat)
    affine = nib.load(t1_path).affine
    nib.Nifti1Image(local_corr_t1_tstat, affine).to_filename(str(DATA_DIR / "local_corr_t1_tstat.nii.gz"))
    nib.Nifti1Image(local_corr_t2_tstat, affine).to_filename(str(DATA_DIR / "local_corr_t2_tstat.nii.gz"))

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
    # compute the difference in correlation between the two pipelines
    df["difference_corr_t1"] = df["corr_t1_medic"] - df["corr_t1_topup"]
    df["difference_corr_t2"] = df["corr_t2_medic"] - df["corr_t2_topup"]
    df["difference_grad_corr_t1"] = df["grad_corr_t1_medic"] - df["grad_corr_t1_topup"]
    df["difference_grad_corr_t2"] = df["grad_corr_t2_medic"] - df["grad_corr_t2_topup"]
    df["difference_nmi_t1"] = df["nmi_t1_medic"] - df["nmi_t1_topup"]
    df["difference_nmi_t2"] = df["nmi_t2_medic"] - df["nmi_t2_topup"]
    df["difference_local_corr_mean_t1"] = df["local_corr_mean_t1_medic"] - df["local_corr_mean_t1_topup"]
    df["difference_local_corr_mean_t2"] = df["local_corr_mean_t2_medic"] - df["local_corr_mean_t2_topup"]
    df["difference_ROC_gw"] = df["roc_gw_medic"] - df["roc_gw_topup"]
    df["difference_ROC_ie"] = df["roc_ie_medic"] - df["roc_ie_topup"]
    df["difference_ROC_vw"] = df["roc_vw_medic"] - df["roc_vw_topup"]
    df["difference_ROC_cb_ie"] = df["roc_cb_ie_medic"] - df["roc_cb_ie_topup"]
    print(df)
    df.to_csv(str(DATA_DIR / "alignment_metrics.csv"), index=False)
    # compute t-tests
    print("T-tests")
    print("T1 Corr:")
    print(ttest_rel(df["corr_t1_medic"], df["corr_t1_topup"]))
    print(ttest_rel(df["corr_t1_medic"], df["corr_t1_topup"]).pvalue < 0.05)
    print("T2 Corr:")
    print(ttest_rel(df["corr_t2_medic"], df["corr_t2_topup"]))
    print(ttest_rel(df["corr_t2_medic"], df["corr_t2_topup"]).pvalue < 0.05)
    print("T1 Grad Corr:")
    print(ttest_rel(df["grad_corr_t1_medic"], df["grad_corr_t1_topup"]))
    print(ttest_rel(df["grad_corr_t1_medic"], df["grad_corr_t1_topup"]).pvalue < 0.05)
    print("T2 Grad Corr:")
    print(ttest_rel(df["grad_corr_t2_medic"], df["grad_corr_t2_topup"]))
    print(ttest_rel(df["grad_corr_t2_medic"], df["grad_corr_t2_topup"]).pvalue < 0.05)
    print("T1 NMI:")
    print(ttest_rel(df["nmi_t1_medic"], df["nmi_t1_topup"]))
    print(ttest_rel(df["nmi_t1_medic"], df["nmi_t1_topup"]).pvalue < 0.05)
    print("T2 NMI:")
    print(ttest_rel(df["nmi_t2_medic"], df["nmi_t2_topup"]))
    print(ttest_rel(df["nmi_t2_medic"], df["nmi_t2_topup"]).pvalue < 0.05)
    print("T1 Local Corr:")
    print(ttest_rel(df["local_corr_mean_t1_medic"], df["local_corr_mean_t1_topup"]))
    print(ttest_rel(df["local_corr_mean_t1_medic"], df["local_corr_mean_t1_topup"]).pvalue < 0.05)
    print("T2 Local Corr:")
    print(ttest_rel(df["local_corr_mean_t2_medic"], df["local_corr_mean_t2_topup"]))
    print(ttest_rel(df["local_corr_mean_t2_medic"], df["local_corr_mean_t2_topup"]).pvalue < 0.05)
    print("ROC GW:")
    print(ttest_rel(df["roc_gw_medic"], df["roc_gw_topup"]))
    print(ttest_rel(df["roc_gw_medic"], df["roc_gw_topup"]).pvalue < 0.05)
    print("ROC IE:")
    print(ttest_rel(df["roc_ie_medic"], df["roc_ie_topup"]))
    print(ttest_rel(df["roc_ie_medic"], df["roc_ie_topup"]).pvalue < 0.05)
    print("ROC VW:")
    print(ttest_rel(df["roc_vw_medic"], df["roc_vw_topup"]))
    print(ttest_rel(df["roc_vw_medic"], df["roc_vw_topup"]).pvalue < 0.05)
    print("ROC CB IE:")
    print(ttest_rel(df["roc_cb_ie_medic"], df["roc_cb_ie_topup"]))
    print(ttest_rel(df["roc_cb_ie_medic"], df["roc_cb_ie_topup"]).pvalue < 0.05)
    from IPython import embed

    embed()
