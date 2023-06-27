import nibabel as nib
import numpy as np
from hdf5storage import loadmat
from IPython import embed
from medic_analysis.common import figures


def main():
    all_struct_mat = loadmat("/home/usr/vana/GMT/David/MEDIC/AllDataStructFull.mat")
    winner_take_all_t1 = np.zeros((91, 109, 91))
    winner_take_all_t2 = np.zeros((91, 109, 91))
    # loop over subjects
    for subject in all_struct_mat["AllDataStruct"][0]:
        for session in subject["Sessions"]["Pipeline"][0]:
            # get T1 spotlight data
            MEDIC_t1 = np.squeeze(session["MEDIC"]["T1Spotlight"])
            TOPUP_t1 = np.squeeze(session["TOPUP"]["T1Spotlight"])
            MEDIC_t1[np.isnan(MEDIC_t1)] = 0
            TOPUP_t1[np.isnan(TOPUP_t1)] = 0
            # compare MEDIC and TOPUP
            if len(MEDIC_t1.shape) == 4:
                MEDIC_win_t1 = np.sum(MEDIC_t1 > TOPUP_t1, axis=3)
                TOPUP_win_t1 = np.sum(TOPUP_t1 > MEDIC_t1, axis=3)
            else:
                MEDIC_win_t1 = (MEDIC_t1 > TOPUP_t1).astype(int)
                TOPUP_win_t1 = (TOPUP_t1 > MEDIC_t1).astype(int)
            winner_take_all_t1 += MEDIC_win_t1 - TOPUP_win_t1

            # get T2 spotlight data
            MEDIC_t2 = np.squeeze(session["MEDIC"]["T2Spotlight"])
            TOPUP_t2 = np.squeeze(session["TOPUP"]["T2Spotlight"])
            MEDIC_t2[np.isnan(MEDIC_t2)] = 0
            TOPUP_t2[np.isnan(TOPUP_t2)] = 0
            breakpoint()
            # compare MEDIC and TOPUP
            if len(MEDIC_t2.shape) == 4:
                MEDIC_win_t2 = np.sum(MEDIC_t2 > TOPUP_t2, axis=3)
                TOPUP_win_t2 = np.sum(TOPUP_t2 > MEDIC_t2, axis=3)
            else:
                MEDIC_win_t2 = (MEDIC_t2 > TOPUP_t2).astype(int)
                TOPUP_win_t2 = (TOPUP_t2 > MEDIC_t2).astype(int)
            winner_take_all_t2 += MEDIC_win_t2 - TOPUP_win_t2
    # threshold images
    # winner_take_all_t1[winner_take_all_t1 > 0] = 50
    # winner_take_all_t1[winner_take_all_t1 < 0] = -50
    # winner_take_all_t2[winner_take_all_t2 > 0] = 50
    # winner_take_all_t2[winner_take_all_t2 < 0] = -50
    # f = figures.data_plotter([winner_take_all_t1, winner_take_all_t2], vmin=-50, vmax=50)
    # sbs = f.get_axes()
    # sbs[1].set_title("T1 Spotlight", loc="center", y=-0.4)
    # sbs[4].set_title("T2 Spotlight", loc="center", y=-0.4)
    # figures.plt.show()
    # nib.Nifti1Image(winner_take_all_t1, np.eye(4)).to_filename("winner_take_all_t1.nii.gz")
    # nib.Nifti1Image(winner_take_all_t2, np.eye(4)).to_filename("winner_take_all_t2.nii.gz")
