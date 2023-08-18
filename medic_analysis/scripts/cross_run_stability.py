from memori.pathman import PathManager as PathMan
import numpy as np
import nibabel as nib
from IPython import embed

ASD_ADHD_DATA = PathMan("/home/usr/vana/Daenerys/ASD_ADHD/NP1173/derivatives/me_pipeline2")


def pairwise_correlation(A, B):
    am = A - np.mean(A, axis=0, keepdims=True)
    bm = B - np.mean(B, axis=0, keepdims=True)
    return (
        am.T
        @ bm
        / (np.sqrt(np.sum(am**2, axis=0, keepdims=True)).T * np.sqrt(np.sum(bm**2, axis=0, keepdims=True)))
    )


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
                medic_cifti_session.append(nib.load(medic_cifti_file))
                pepolar_cifti_session.append(nib.load(pepolar_cifti_file))
        print("Number of runs:", len(medic_cifti_session))
        # now grab the shape of the dconn files
        shape = medic_cifti_session[0].shape[0]

        # create lower triangle index
        indices = np.tril_indices(shape)

        # for each run, grab the lower triangle
        medic_cifti_session = [i.get_fdata()[indices] for i in medic_cifti_session[0:1]]
        pepolar_cifti_session = [i.get_fdata()[indices] for i in pepolar_cifti_session[0:1]]
        breakpoint()
        # compute correlation between runs
        medic_corrs = []
        for n, i in enumerate(medic_cifti_session):
            for m, j in enumerate(medic_cifti_session):
                if m <= n:
                    continue
                medic_corrs.append(pairwise_correlation(i, j))
                breakpoint()
                break
            break
        pepolar_corrs = []
        # for n, i in enumerate(pepolar_cifti_session):
        #     for m, j in enumerate(pepolar_cifti_session):
        #         if m <= n:
        #             continue
        #         pepolar_corrs.append(np.corrcoef(i.dataobj, j.dataobj)[0, 1])
        subject_corrs[subject.name] = {"medic": medic_corrs, "pepolar": pepolar_corrs}
        break
