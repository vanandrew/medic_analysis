import numpy as np
from skimage.morphology import ball, binary_dilation
from scipy.integrate import cumulative_trapezoid
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
            "-cost",
            "leastsquares",
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


def compute_ROC(true_positive_index, false_positive_index, data, number_of_divisions):
    true_positives = data[true_positive_index]
    false_positives = data[false_positive_index]

    data_min = np.min(data)
    data_max = np.max(data)

    true_positive_array, bins = np.histogram(true_positives, bins=number_of_divisions, range=(data_min, data_max))
    false_positive_array, _ = np.histogram(false_positives, bins=number_of_divisions, range=(data_min, data_max))

    # get bin centers
    data_axis = np.mean(np.vstack([bins[0:-1], bins[1:]]), axis=0)

    true_positive_array = true_positive_array / np.trapz(true_positive_array, data_axis)
    false_positive_array = false_positive_array / np.trapz(false_positive_array, data_axis)

    cum_true_positive = 1 - cumulative_trapezoid(true_positive_array, data_axis)
    cum_true_positive[cum_true_positive < 0] = 0

    cum_false_positive = 1 - cumulative_trapezoid(false_positive_array, data_axis)
    cum_false_positive[cum_false_positive < 0] = 0

    auc = -np.trapz(cum_true_positive, cum_false_positive)

    return auc


def roc_metrics(image, parcel):
    strel = ball(1)

    # get cerebellum
    cerebellum = np.isin(parcel, [8, 47, 7, 46])
    cerebellum_shell = binary_dilation(cerebellum, strel) & ~cerebellum
    cerebellum_and_shell = cerebellum | cerebellum_shell

    # get gray matter and non gray matter
    gray_matter = ((parcel >= 1000) & (parcel <= 3000)) | np.isin(parcel, [47, 8, 51, 52, 12, 13, 49, 10, 16])
    non_gray_matter = ~gray_matter & (parcel != 0)

    # get shell and shell + gray matter
    shell = binary_dilation(parcel != 0, strel) & ~(parcel != 0)
    shell_and_gray_matter = gray_matter | shell

    # get ventricles, ventricles + shell, and ventricles shell
    ventricles = np.isin(parcel, [4, 43])
    ventricles_and_shell = binary_dilation(ventricles, strel)
    ventricles_shell = ventricles_and_shell & ~ventricles

    # compute ROCs along boundaries
    ROC_gw = compute_ROC(gray_matter[parcel != 0], non_gray_matter[parcel != 0], image[parcel != 0], 100)
    ROC_ie = compute_ROC(
        gray_matter[shell_and_gray_matter],
        shell[shell_and_gray_matter],
        image[shell_and_gray_matter],
        100,
    )
    ROC_vw = compute_ROC(
        ventricles[ventricles_and_shell],
        ventricles_shell[ventricles_and_shell],
        image[ventricles_and_shell],
        25,
    )
    ROC_cb_ie = compute_ROC(
        cerebellum[cerebellum_and_shell],
        cerebellum_shell[cerebellum_and_shell],
        image[cerebellum_and_shell],
        25,
    )
    return ROC_gw, ROC_ie, ROC_vw, ROC_cb_ie
