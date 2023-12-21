"""Perform respiration analysis."""
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.signal as ss
from memori.helpers import working_directory
from memori.pathman import PathManager as PathMan

from medic_analysis.common.figures import data_plotter, plt, sns

from . import parser

# Define the path to the BIDS dataset
BIDS_DATA_DIR = "/home/usr/vana/GMT2/Andrew/RESPTEST"


def main():
    # add to parser
    parser.add_argument("--runs", nargs="+", default=["02"])
    parser.add_argument("--num_frames", type=int, default=510, help="number of frames in the scan")
    parser.add_argument(
        "--plot_only",
        nargs="+",
        type=int,
        help="List of figures to plot.",
        default=None,
    )

    # call the parser
    args = parser.parse_args()

    # if bids dir not specified, use default
    if args.bids_dir is None:
        args.bids_dir = BIDS_DATA_DIR

    # set output dir
    if args.output_dir is None:
        args.output_dir = (PathMan(args.bids_dir) / "derivatives").path

    # make the output dir if not exist
    output_dir = PathMan(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.plot_only is None or 0 in args.plot_only:
        # load the run 2 data
        tSNR_MEDIC = (
            output_dir
            / "me_pipeline"
            / "sub-MSCHD02"
            / "ses-resptest"
            / "bold2"
            / "sub-MSCHD02_b2_faln_xr3d_uwrp_on_MNI152_T1_2mm_Swgt_norm_SNR.nii.gz"
        )
        tSNR_MEDIC = nib.load(tSNR_MEDIC.path).get_fdata()
        tSNR_TOPUP = (
            output_dir
            / "me_pipeline"
            / "sub-MSCHD02"
            / "ses-resptestwTOPUP"
            / "bold2"
            / "sub-MSCHD02_b2_faln_xr3d_uwrp_on_MNI152_T1_2mm_Swgt_norm_SNR.nii.gz"
        )
        tSNR_TOPUP = nib.load(tSNR_TOPUP.path).get_fdata()
        f0 = plt.figure(figsize=(10, 8), layout="constrained")
        fig = data_plotter(
            [tSNR_TOPUP, tSNR_MEDIC, tSNR_MEDIC - tSNR_TOPUP],
            colormaps=["rocket", "rocket", "icefire"],
            vmin=[0, 0, -50],
            vmax=[150, 150, 50],
            colorbar=True,
            colorbar_label="tSNR",
            colorbar2=True,
            colorbar2_label="tSNR Difference",
            colorbar2_source_idx=(2, 0),
            slices=(55, 55, 32),
            figure=f0,
        )
        sbs = fig.get_axes()
        sbs[1].set_title("(A) tSNR (TOPUP)", loc="center", y=-0.3)
        sbs[4].set_title("(B) tSNR (MEDIC)", loc="center", y=-0.3)
        sbs[7].set_title("(C) tSNR Difference (MEDIC - TOPUP)", loc="center", y=-0.3)
        plt.show()

    if args.plot_only is None or 1 in args.plot_only:
        # loop over runs for processing
        for run in args.runs:
            (output_dir / f"run-{run}").mkdir(exist_ok=True)
            with working_directory((output_dir / f"run-{run}").path):
                # get the motion parameters
                motion_params_premedic = np.loadtxt("mcflirt.par")
                motion_params_premedic[:, :3] = np.rad2deg(motion_params_premedic[:, :3])
                motion_params_premedic[:, :3] = 50 * (np.pi / 180) * motion_params_premedic[:, :3]
                motion_params_premedic = motion_params_premedic[: args.num_frames]
                motion_params_postmedic = np.loadtxt("mcflirt_corrected.par")
                motion_params_postmedic[:, :3] = np.rad2deg(motion_params_postmedic[:, :3])
                motion_params_postmedic[:, :3] = 50 * (np.pi / 180) * motion_params_postmedic[:, :3]
                motion_params_postmedic = motion_params_postmedic[: args.num_frames]

                # convert to dataframes
                motion_params_premedic = pd.DataFrame(motion_params_premedic, columns=["rx", "ry", "rz", "x", "y", "z"])
                motion_params_postmedic = pd.DataFrame(
                    motion_params_postmedic, columns=["rx", "ry", "rz", "x", "y", "z"]
                )

                # plot the motion parameters
                f = plt.figure(figsize=(16, 8), layout="constrained")
                axes = f.subplots(2, 3)
                for i, col in enumerate(motion_params_premedic.columns):
                    ax = sns.lineplot(
                        x=np.arange(args.num_frames) * 1.761,
                        y=motion_params_premedic[col],
                        ax=axes[i // 3, i % 3],  # type: ignore
                    )
                    ax.set(xlabel="Time (s)", ylabel="Displacement (mm)", title=col)
                    ax = sns.lineplot(
                        x=np.arange(args.num_frames) * 1.761,
                        y=motion_params_postmedic[col],
                        ax=axes[i // 3, i % 3],  # type: ignore
                    )
                    ax.set(xlabel="Time (s)", ylabel="Displacement (mm)", title=col)
                    ax.legend(["pre-medic", "post-medic"])

                # compute spectral power for each motion parameter
                f = plt.figure(figsize=(16, 8), layout="constrained")
                axes = f.subplots(2, 3)
                skip = 0
                for i, col in enumerate(motion_params_premedic.columns):
                    # get the power for each motion parameteradd_subplot(2, 3)
                    freqs, power_premedic = ss.welch(motion_params_premedic[col], fs=1 / 1.761, detrend="linear")
                    freqs, power_postmedic = ss.welch(motion_params_postmedic[col], fs=1 / 1.761, detrend="linear")

                    # plot the power
                    ax = sns.lineplot(x=freqs[skip:], y=power_premedic[skip:], ax=axes[i // 3, i % 3])  # type: ignore
                    ax.set(xlabel="Frequency (Hz)", ylabel="Power", title=col, ylim=(0, 0.1))
                    ax = sns.lineplot(x=freqs[skip:], y=power_postmedic[skip:], ax=axes[i // 3, i % 3])  # type: ignore
                    ax.set(xlabel="Frequency (Hz)", ylabel="Power", title=col, ylim=(0, 0.1))
                    ax.legend(["pre-medic", "post-medic"])

                # compute FD for pre-medic and post-medic
                fd_premedic = np.sum(np.abs(motion_params_premedic.diff()), axis=1)
                fd_postmedic = np.sum(np.abs(motion_params_postmedic.diff()), axis=1)

                # plot FD
                f = plt.figure(figsize=(16, 8), layout="constrained")
                axes = f.subplots(1, 1)
                ax = sns.lineplot(x=np.arange(args.num_frames) * 1.761, y=fd_premedic, ax=axes)  # type: ignore
                ax.set(xlabel="Time (s)", ylabel="FD", title="FD")
                ax = sns.lineplot(x=np.arange(args.num_frames) * 1.761, y=fd_postmedic, ax=axes)  # type: ignore
                ax.set(xlabel="Time (s)", ylabel="FD", title="FD")
                ax.legend(["pre-medic", "post-medic"])

                plt.show()
                breakpoint()
