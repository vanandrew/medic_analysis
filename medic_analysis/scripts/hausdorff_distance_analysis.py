import nibabel as nib
import numpy as np
from scipy.optimize import minimize_scalar, brute
from scipy.stats import ttest_rel
from warpkit.utilities import compute_hausdorff_distance
from warpkit.unwrap import get_largest_connected_component
from skimage.morphology import dilation, ball, binary_dilation
from sklearn.metrics import accuracy_score
from pathlib import Path
import simplebrainviewer as sbv
import pandas as pd
from omni.interfaces.ants import N4BiasFieldCorrection
from tempfile import TemporaryDirectory
from . import DATA_DIR

DATA_OUTPUT_DIR = "/home/usr/vana/Daenerys/ASD_ADHD/NP1173/derivatives/me_pipeline2"
# DATA_OUTPUT_DIR = "/home/usr/vana/GMT2/Andrew/UPenn/derivatives/me_pipeline"


def normalize(data):
    # data = zscore(data, axis=None, nan_policy="omit")
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def main():
    # make list to store the results
    results = []

    # loop over subjects
    for subject_dir in sorted(Path(DATA_OUTPUT_DIR).glob("sub-*")):
        # if "sub-20037" not in subject_dir.name:
        #     continue
        # load the subject's wmparc file
        try:
            wmparc_path = next((subject_dir / "T1" / "atlas").glob("*_wmparc_on_MNI152_T1_2mm.nii.gz"))
        except StopIteration:
            continue
        wmparc_img = nib.load(wmparc_path)

        # get the mask data
        mask_data = wmparc_img.get_fdata().squeeze()

        # get gray matter and non gray matter
        # gray_mask = (mask_data != 0)
        gray_mask = ((mask_data >= 1000) & (mask_data <= 3000)) | np.isin(
            mask_data, [47, 8, 51, 52, 12, 13, 49, 10, 16]
        )
        white_mask = (mask_data != 0) & ~gray_mask
        gray_mask_img = nib.Nifti1Image(gray_mask.astype("f8"), wmparc_img.affine)

        # get the gray matter mask
        # gray_mask = ((mask_data >= 2000) & (mask_data <= 2035)) | ((mask_data >= 1000) & (mask_data <= 1035))
        # # get the white matter mask
        # white_mask = ((mask_data >= 3000) & (mask_data <= 3035)) | ((mask_data >= 4000) & (mask_data <= 4035))
        # # make into images
        # gray_mask_img = nib.Nifti1Image(gray_mask.astype("f8"), wmparc_img.affine)
        # gray_mask_img.to_filename(f"{subject_dir.name}_gray_mask.nii.gz")

        # initialize zeros data to place masked data
        zeros_data = np.zeros(mask_data.shape)

        # loop over the sessions for the subject
        for session_dir in sorted(subject_dir.glob("ses-*")):
            isTOPUP = False
            session_name = session_dir.name
            if "wTOPUP" in session_dir.name:
                isTOPUP = True
                session_name = session_dir.name.split("w")[0]
            if "GRE" in session_dir.name:
                continue
            # loop over each bold run
            for bold_dir in sorted(session_dir.glob("bold?")):
                label = "TOPUP" if isTOPUP else "MEDIC"
                print(f"Processing {subject_dir.name} {session_name} {bold_dir.name} {label}")
                bold_path = next(bold_dir.glob("*_faln_xr3d_uwrp_on_MNI152_T1_2mm_Swgt_norm.nii"))
                tmp_dir = TemporaryDirectory()
                bold_img = nib.load(bold_path)
                bold_data = np.squeeze(bold_img.dataobj[..., 0])
                nib.Nifti1Image(bold_data, bold_img.affine).to_filename(Path(tmp_dir.name) / "tmp.nii.gz")
                N4BiasFieldCorrection(
                    out_file=str(Path(tmp_dir.name) / "tmp2.nii.gz"),
                    in_file=str(Path(tmp_dir.name) / "tmp.nii.gz"),
                )
                bold_img = nib.load(str(Path(tmp_dir.name) / "tmp2.nii.gz"))
                bold_data = bold_img.get_fdata()

                # grab the voxels for this image
                func_voxels = bold_data[gray_mask | white_mask]

                # create functional data to generate masks on
                func_graywhite = zeros_data.copy()
                func_graywhite[gray_mask | white_mask] = func_voxels
                func_graywhite = normalize(func_graywhite)

                # write function to compute hausdorff distance and threshold
                def threshold_and_compute_hausdorff_distance(anat_labels, func, threshold):
                    # threshold the voxels
                    func_labels_data = func > threshold

                    # only grab contiguous voxels
                    # func_labels_data = get_largest_connected_component(func_labels_data)
                    func_labels = nib.Nifti1Image(func_labels_data.astype("f8"), anat_labels.affine)

                    # compute the hausdorff distance
                    try:
                        return compute_hausdorff_distance(anat_labels, func_labels)
                    except RuntimeError:
                        return 999999

                # optimize the threshold to compute the minimum hausdorff distance
                def gray_threshold_func(gray_threshold):
                    return threshold_and_compute_hausdorff_distance(gray_mask_img, func_graywhite, gray_threshold)

                def threshold_and_compute_accuracy(anat_labels, func, threshold):
                    # threshold the voxels
                    func_labels_data = func > threshold
                    # only grab contiguous voxels

                    # compute the hausdorff distance
                    return -accuracy_score(anat_labels, func_labels_data)

                def accuracy_func(gray_threshold):
                    return threshold_and_compute_accuracy(
                        gray_mask[gray_mask | white_mask],
                        normalize(func_graywhite[gray_mask | white_mask]),
                        gray_threshold,
                    )

                best_threshold, obj_value, _, _ = brute(
                    gray_threshold_func,
                    ranges=[(0.4, 1.0)],
                    Ns=200,
                    workers=1,
                    full_output=True,
                )
                threshold_value = best_threshold[0]
                # second pass threshold +/- 0.05
                best_threshold, obj_value, _, _ = brute(
                    gray_threshold_func,
                    ranges=[(threshold_value - 0.05, threshold_value + 0.05)],
                    Ns=200,
                    workers=1,
                    full_output=True,
                )

                obj_value = compute_hausdorff_distance(
                    gray_mask_img,
                    nib.Nifti1Image((func_graywhite > threshold_value).astype("f8"), gray_mask_img.affine),
                )
                print(obj_value)
                if isTOPUP:
                    results.append(
                        {
                            "subject": subject_dir.name,
                            "session": session_name,
                            "bold": bold_dir.name,
                            "label": label,
                            "gray_hausdorff_topup": obj_value,
                            "gray_threshold_topup": threshold_value,
                        }
                    )
                else:
                    results.append(
                        {
                            "subject": subject_dir.name,
                            "session": session_name,
                            "bold": bold_dir.name,
                            "label": label,
                            "gray_hausdorff_medic": obj_value,
                            "gray_threshold_medic": threshold_value,
                        }
                    )
                # nib.Nifti1Image((func_graywhite > threshold_value).astype("f8"), bold_img.affine).to_filename(
                #     f"{subject_dir.name}_{session_name}_{bold_dir.name}_{label}_gray.nii.gz"
                # )
                tmp_dir.cleanup()

    # turn results into a dataframe
    results_df = pd.DataFrame(results)
    # separate the results into medic and topup
    medic_results_df = results_df[results_df["label"] == "MEDIC"].drop(
        columns=["gray_threshold_topup", "gray_hausdorff_topup", "label"]
    )
    topup_results_df = results_df[results_df["label"] == "TOPUP"].drop(
        columns=["gray_threshold_medic", "gray_hausdorff_medic", "label"]
    )
    # merge the results
    results_df = pd.merge(medic_results_df, topup_results_df, on=["subject", "session", "bold"])
    # who wins?
    results_df["gray_hausdorff_distance"] = results_df["gray_hausdorff_medic"] - results_df["gray_hausdorff_topup"]

    print(results_df)
    # save the results
    results_df.to_csv(str(DATA_DIR / "hausdorff_distance_results.csv"), index=False)

    print(ttest_rel(results_df["gray_hausdorff_medic"], results_df["gray_hausdorff_topup"]))
    from IPython import embed

    embed()
