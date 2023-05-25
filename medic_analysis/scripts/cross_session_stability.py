from memori.pathman import PathManager as PathMan
import numpy as np
import nibabel as nib
from IPython import embed

ASD_ADHD_DATA = PathMan("/home/usr/vana/Daenerys/ASD_ADHD/NP1173/derivatives/me_pipeline")


def main():
    # load the path to ASD_ADHD data
    dataset = ASD_ADHD_DATA

    # loop through subjects
    subject_corrs = {}
    for subject in dataset.glob("sub-*"):
        print(f"Processing {subject.name}")
        # loop through sessions
        medic_cifti_session = []
        pepolar_cifti_session = []
        for session in subject.glob("ses-*"):
            if "wTOPUP" in session.name:
                continue
            # loop through cifti files in session
            for medic_cifti_file in session.glob("cifti_correlation/*.dconn.nii"):
                pepolar_cifti_file = pepolar_cifti_file = (
                    PathMan(session.path + "wTOPUP") / "cifti_correlation" / medic_cifti_file.name
                )
                medic_cifti = nib.load(medic_cifti_file)
                pepolar_cifti = nib.load(pepolar_cifti_file)
                medic_cifti_session.append(medic_cifti)
                pepolar_cifti_session.append(pepolar_cifti)
        print("Number of runs:", len(medic_cifti_session))
        # compute correlation between runs
        medic_corrs = []
        for n, i in enumerate(medic_cifti_session):
            for m, j in enumerate(medic_cifti_session):
                if m <= n:
                    continue
                medic_corrs.append(np.corrcoef(i.dataobj, j.dataobj)[0, 1])
        pepolar_corrs = []
        for n, i in enumerate(pepolar_cifti_session):
            for m, j in enumerate(pepolar_cifti_session):
                if m <= n:
                    continue
                pepolar_corrs.append(np.corrcoef(i.dataobj, j.dataobj)[0, 1])
        subject_corrs[subject.name] = {"medic": medic_corrs, "pepolar": pepolar_corrs}
        breakpoint()
