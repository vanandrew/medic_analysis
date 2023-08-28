from pathlib import Path
import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes
from warpkit.utilities import get_x_orient_transform


ME_PIPELINE_PATH = Path("/data/Daenerys/ASD_ADHD/NP1173/derivatives/me_pipeline2/")
ME_PIPELINE_PATH = Path("/home/usr/vana/GMT2/Andrew/UMinn/derivatives/")
ME_PIPELINE_PATH = Path("/home/usr/vana/GMT2/Andrew/UPenn/derivatives/me_pipeline")
ME_PIPELINE_PATH = Path("/home/usr/vana/GMT2/Andrew/SLICETEST/derivatives/me_pipeline")


def main():
    # create metadata
    meta1 = {
        "AnatomicalStructurePrimary": "Cerebellum",
        "AnatomicalStructureSecondary": "Pial",
        "GeometricType": "Anatomical",
        "Name": "#1",
        "configuration_id": "Unknown",
    }
    meta2 = {
        "AnatomicalStructurePrimary": "CerebellarWhiteMatterLeft",
        "AnatomicalStructureSecondary": "Pial",
        "GeometricType": "Anatomical",
        "Name": "#1",
        "configuration_id": "Unknown",
    }
    meta3 = {
        "Name": "#2",
        "TopologicalType": "Closed",
        "UniqueID": "{e2ef4622-4b06-4612-9159-1d0b9802684a}",
    }

    # loop over subjects
    for subject in sorted(ME_PIPELINE_PATH.glob("sub-*")):
        subject_name = subject.name
        print(subject_name)
        atlas_dir = subject / "T1" / "atlas"
        wm_parc_path = atlas_dir / f"{subject_name}_wmparc_on_MNI152_T1_2mm.nii.gz"

        # load the wmparc
        wm_parc = nib.load(wm_parc_path)
        data = wm_parc.get_fdata()
        # wm_parc.to_filename("/home/usr/vana/GMT2/Andrew/test.nii.gz")

        # get cerebellum
        cerebellum = (data == 8) | (data == 47) | (data == 7) | (data == 46)
        vertices, faces, _, _ = marching_cubes(cerebellum)
        # project vertices to scanner space
        affine = wm_parc.affine.copy()
        # affine[0, 0] *= -1
        # affine[0, 3] = (data.shape[0] - 1) * np.abs(affine[0, 0])
        # affine[1, 3] = 0
        # affine[2, 3] = 0
        vertices = nib.affines.apply_affine(affine, vertices)
        gifti = nib.gifti.GiftiImage()
        gifti.add_gifti_data_array(
            nib.gifti.GiftiDataArray(
                data=vertices,
                intent="NIFTI_INTENT_POINTSET",
                datatype="NIFTI_TYPE_FLOAT32",
                coordsys=nib.gifti.GiftiCoordSystem("NIFTI_XFORM_TALAIRACH", "NIFTI_XFORM_TALAIRACH", np.eye(4)),
                meta=meta1,
            )
        )
        gifti.add_gifti_data_array(
            nib.gifti.GiftiDataArray(
                data=faces,
                intent="NIFTI_INTENT_TRIANGLE",
                datatype="NIFTI_TYPE_FLOAT32",
                coordsys=nib.gifti.GiftiCoordSystem("NIFTI_XFORM_TALAIRACH", "NIFTI_XFORM_TALAIRACH", np.eye(4)),
                meta=meta3,
            )
        )
        gifti.to_filename(atlas_dir / f"{subject_name}_cerebellum_on_MNI152_T1_2mm.surf.gii")

        # get cerebellum white matter
        cerebellum = (data == 8) | (data == 47)
        vertices, faces, _, _ = marching_cubes(cerebellum)
        # project vertices to scanner space
        affine = wm_parc.affine.copy()
        # affine[0, 0] *= -1
        # affine[0, 3] = (data.shape[0] - 1) * np.abs(affine[0, 0])
        # affine[1, 3] = 0
        # affine[2, 3] = 0
        vertices = nib.affines.apply_affine(affine, vertices)
        gifti = nib.gifti.GiftiImage()
        gifti.add_gifti_data_array(
            nib.gifti.GiftiDataArray(
                data=vertices,
                intent="NIFTI_INTENT_POINTSET",
                datatype="NIFTI_TYPE_FLOAT32",
                coordsys=nib.gifti.GiftiCoordSystem("NIFTI_XFORM_TALAIRACH", "NIFTI_XFORM_TALAIRACH", np.eye(4)),
                meta=meta2,
            )
        )
        gifti.add_gifti_data_array(
            nib.gifti.GiftiDataArray(
                data=faces,
                intent="NIFTI_INTENT_TRIANGLE",
                datatype="NIFTI_TYPE_FLOAT32",
                coordsys=nib.gifti.GiftiCoordSystem("NIFTI_XFORM_TALAIRACH", "NIFTI_XFORM_TALAIRACH", np.eye(4)),
                meta=meta3,
            )
        )
        gifti.to_filename(atlas_dir / f"{subject_name}_cerebellum_white_matter_on_MNI152_T1_2mm.surf.gii")
