#!/usr/bin/env python3
import nibabel as nib
import numpy as np
from nibabel.cifti2.cifti2_axes import BrainModelAxis, ScalarAxis
from nibabel.cifti2.cifti2 import Cifti2Image

# load all runs
img = nib.load("Similarity_toABCDavg_MEDICandTOPUP_allruns.dscalar.nii")

# get indices
medic_idx = np.array([not "TOPUP" in i for i in img.header.get_axis(0).name])
topup_idx = np.array(["TOPUP" in i for i in img.header.get_axis(0).name])

# split the data
data = img.get_fdata()
medic_data = data[medic_idx, :59412]
topup_data = data[topup_idx, :59412]

# get labels
labels = np.array(
    [
        (i.split("sub-")[1].split("/")[0], i.split("ses-")[1].split("/")[0], i.split("_b")[1].split("_")[0])
        for i in img.header.get_axis(0).name
        if "TOPUP" not in i
    ]
)

# create new scalar axis
new_scalar_axis = ScalarAxis(["_".join(i) for i in labels])

# create new brain model axis containing only cortex
axes = None
i = 0
for name, slc, axis in img.header.get_axis(1).iter_structures():
    if axes is None:
        axes = axis
    else:
        axes += axis
    if i == 1:
        break
    i += 1

# create new images
medic_img = nib.cifti2.cifti2.Cifti2Image(medic_data, [new_scalar_axis, axes], img.nifti_header)
topup_img = nib.cifti2.cifti2.Cifti2Image(topup_data, [new_scalar_axis, axes], img.nifti_header)
medic_img.to_filename("Similarity_toABCDavg_MEDIC_allruns.dscalar.nii")
topup_img.to_filename("Similarity_toABCDavg_TOPUP_allruns.dscalar.nii")
