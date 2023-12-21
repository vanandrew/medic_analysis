import logging
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import seaborn as sns
from bids import BIDSLayout
from memori.helpers import working_directory
from memori.pathman import PathManager as PathMan
from scipy.signal import filtfilt, iirfilter, periodogram
from warpkit.distortion import medic
from warpkit.unwrap import create_brain_mask

from medic_analysis.common.align import apply_framewise_mats, framewise_align
from medic_analysis.common.figures import plt, sns

from . import DATA_DIR, FIGURES_DIR, parser

sns.set_theme(style="darkgrid", palette="pastel", font="Satoshi")


# Define the path to the BIDS dataset
BIDS_DATA_DIR = "/home/usr/vana/GMT2/Andrew/RESPTEST"


def parse_log_lines(physio_log_lines: List[str]) -> Dict:
    # initialize dictionary
    data = dict()
    # physio data
    data["ACQ_TIME_TIC"] = []
    data["VALUE"] = []
    # for acquisition info
    data["VOLUME"] = []
    data["SLICE"] = []
    data["ECHO"] = []
    data["ACQ_START_TICS"] = []
    data["ACQ_FINISH_TICS"] = []

    # loop through physio log lines
    for line in physio_log_lines:
        # header information
        if "UUID" in line:
            data["UUID"] = line.split()[2]
        elif "ScanDate" in line:
            data["ScanDate"] = line.split()[2]
        elif "LogVersion" in line:
            data["LogVersion"] = line.split()[2]
        elif "LogDataType" in line:
            data["LogDataType"] = line.split()[2]
        elif "SampleTime" in line:
            data["SampleTime"] = float(line.split()[2])
        elif "NumSlices" in line:
            data["NumSlices"] = int(line.split()[2])
        elif "NumVolumes" in line:
            data["NumVolumes"] = int(line.split()[2])
        elif "NumEchoes" in line:
            data["NumEchoes"] = int(line.split()[2])
        # actual data
        elif any([channel in line for channel in ("PULS", "RESP", "EXT", "ECG")]):
            # split the line
            line_split = line.split()

            # get the acquisition time
            data["ACQ_TIME_TIC"].append(int(line_split[0]))

            # get the value
            data["VALUE"].append(float(line_split[2]))
        # else this might be ACQUISTION_INFO table
        else:
            # check if LogDataType was set to ACQUISITION_INFO
            if data["LogDataType"] == "ACQUISITION_INFO":
                # we want to skip the header line
                if "VOLUME" in line:
                    continue

                # FirstTime/LastTime Lines
                if "FirstTime" in line:
                    data["FirstTime"] = line.split()[2]
                    continue
                elif "LastTime" in line:
                    data["LastTime"] = line.split()[2]
                    continue

                # split the line
                line_split = line.split()

                # make sure line isn't empty
                if len(line_split) == 0:
                    continue

                # store the data
                data["VOLUME"].append(int(line_split[0]))
                data["SLICE"].append(int(line_split[1]))
                data["ACQ_START_TICS"].append(int(line_split[2]))
                data["ACQ_FINISH_TICS"].append(int(line_split[3]))
                data["ECHO"].append(int(line_split[4]))

    # delete the existing ACQ_TIME_TIC and VALUE lists
    # if this was the ACQUISITION_INFO
    if data["LogDataType"] == "ACQUISITION_INFO":
        del data["ACQ_TIME_TIC"]
        del data["VALUE"]
    else:  # else this was just a regular physio log, so delete the ACQUISITION_INFO tables
        del data["VOLUME"]
        del data["SLICE"]
        del data["ACQ_START_TICS"]
        del data["ACQ_FINISH_TICS"]
        del data["ECHO"]

    # return data dictionary
    return data


def waveform_data_to_frame(waveform_data: Dict) -> pd.DataFrame:
    # get a subset of the dictionary that are lists
    subset = {key: waveform_data[key] for key in waveform_data.keys() if isinstance(waveform_data[key], list)}

    # get key/values not in subset
    attrs = {key: waveform_data[key] for key in waveform_data.keys() if key not in subset.keys()}

    # convert to dataframe
    frame = pd.DataFrame(subset)
    frame.attrs = attrs

    # add TIME column for ACQ_TIME_TIC (seconds)
    if "ACQ_TIME_TIC" in frame.columns:
        frame["TIME"] = frame["ACQ_TIME_TIC"] * 2.5 / 1000
        # frame.set_index("ACQ_TIME_TIC", inplace=True)

    # do the same for ACQ_START_TICS and ACQ_FINISH_TICS
    if "ACQ_START_TICS" in frame.columns:
        frame["START_TIME"] = frame["ACQ_START_TICS"] * 2.5 / 1000
    if "ACQ_FINISH_TICS" in frame.columns:
        frame["FINISH_TIME"] = frame["ACQ_FINISH_TICS"] * 2.5 / 1000
    if "START_TIME" in frame.columns and "FINISH_TIME" in frame.columns:
        frame["MEAN_SLICE_TIME"] = frame[["START_TIME", "FINISH_TIME"]].mean(axis=1)
        # make interval index for frame
        interval_index = pd.IntervalIndex.from_arrays(frame["ACQ_START_TICS"], frame["ACQ_FINISH_TICS"], closed="left")
        frame.set_index(interval_index, inplace=True)
        frame.index.name = "interval"

    # return the dataframe
    return frame


def main():
    # add to parser
    parser.add_argument("--runs", nargs="+", default=["02"])

    # call the parser
    args = parser.parse_args()

    # if bids dir not specified, use default
    if args.bids_dir is None:
        args.bids_dir = BIDS_DATA_DIR

    # Load the dataset
    layout = BIDSLayout(args.bids_dir, database_path=args.bids_dir)

    # set output dir
    if args.output_dir is None:
        args.output_dir = (PathMan(args.bids_dir) / "derivatives").path

    # make the output dir if not exist
    output_dir = PathMan(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # get number of runs
    num_runs = len(args.runs)

    # create figure with subfigures
    # fig = plt.figure(figsize=(10 * num_runs, 10 * num_runs), layout="constrained")
    # subfigs = fig.subfigures(num_runs, 2, wspace=0.05, hspace=0.1)

    # loop over runs for processing
    for run_idx, run in enumerate(args.runs):
        (output_dir / f"run-{run}").mkdir(exist_ok=True)
        with working_directory((output_dir / f"run-{run}").path):
            # load image data with params
            mag = layout.get(part="mag", datatype="func", run=run, extension="nii.gz")
            phase = layout.get(part="phase", datatype="func", run=run, extension="nii.gz")
            mag_imgs = [m.get_image() for m in mag]
            phase_imgs = [p.get_image() for p in phase]
            TEs = [m.get_metadata()["EchoTime"] * 1000 for m in mag]
            total_readout_time = mag[0].get_metadata()["TotalReadoutTime"]
            phase_encoding_direction = mag[0].get_metadata()["PhaseEncodingDirection"]

            # get reference frame
            ref_data = mag_imgs[0].dataobj[..., 100]
            nib.Nifti1Image(ref_data, mag_imgs[0].affine).to_filename("ref.nii")

            # compute framewise correction
            if not PathMan("mcflirt.nii.gz").exists():
                framewise_align("ref.nii", mag[0].path, "mcflirt")

            # get num frames
            num_frames = mag_imgs[0].shape[-1] - 3
            print(f"Run {run}")
            print(f"Number of frames: {num_frames}")

            # run medic
            if not PathMan("fmap.nii.gz").exists():
                fmap_native, dmap, fmap = medic(
                    phase_imgs,
                    mag_imgs,
                    TEs,
                    total_readout_time,
                    phase_encoding_direction,
                    n_cpus=8,
                    frames=[i for i in range(num_frames)],
                    border_size=3,
                    svd_filt=10,
                )
                fmap_native.to_filename("fmap_native.nii.gz")
                dmap.to_filename("dmap.nii.gz")
                fmap.to_filename("fmap.nii.gz")

            # apply framewise correction
            if not PathMan("fmap_aligned.nii.gz").exists():
                apply_framewise_mats("ref.nii", "fmap.nii.gz", "mcflirt.mat", "fmap_aligned.nii.gz")

            # make brain mask of reference image
            ref_img = nib.load("ref.nii")
            brain_mask = create_brain_mask(ref_img.get_fdata(), 0)

            # BELOW TAKEN FROM bidsphysio PACKAGE

            # load physio data dicom
            dcm = pydicom.read_file((PathMan(args.bids_dir) / "MB" / f"physio_run-{run}.dcm").path)

            # Extract data from Siemens spectroscopy tag (0x7fe1, 0x1010)
            # Yields one long byte array
            try:
                physio_data = dcm[0x7FE1, 0x1010].value
            except KeyError:
                raise RuntimeError(
                    "Physiological data are not available or cannot be accessed from current input data: "
                    "Element tag [0x7fe1, 0x1010] is missing."
                )

            # get number of points and rows
            n_points = len(physio_data)
            n_rows = dcm.AcquisitionNumber
            if n_points % n_rows:
                raise ValueError("* Points (%d) is not an integer multiple of rows (%d) - exiting" % (n_points, n_rows))

            # get number of columns
            n_cols = int(n_points / n_rows)
            if n_points % 1024:
                raise ValueError("* Columns (%d) is not an integer multiple of 1024 (%d) - exiting" % (n_cols, 1024))

            # get number of waveforms
            n_waves = int(n_cols / 1024)
            wave_len = int(n_points / n_waves)

            # make dictionary to store each waveform
            waveform_data = dict()

            # loop over waveforms
            for wc in range(n_waves):
                # parse waveforms
                logging.info("Parsing waveform %d" % wc)
                offset = wc * wave_len
                wave_data = physio_data[slice(offset, offset + wave_len)]
                data_len = int.from_bytes(wave_data[0:4], byteorder=sys.byteorder)
                fname_len = int.from_bytes(wave_data[4:8], byteorder=sys.byteorder)
                fname = wave_data[slice(8, 8 + fname_len)].decode("utf-8", "ignore")
                logging.info("Data length     : %d" % data_len)
                logging.info("Filename length : %d" % fname_len)
                logging.info("Filename        : %s" % fname)

                # Extract waveform log byte data
                log_bytes = wave_data[slice(1024, 1024 + data_len)]
                # Convert from a bytes literal to a UTF-8 encoded string, ignoring errors; split lines:
                physio_log_lines = log_bytes.decode("utf-8", "ignore").splitlines()

                # Parse physio_log_lines
                data = parse_log_lines(physio_log_lines)

                # store the data with DataType key
                waveform_data[data["LogDataType"]] = waveform_data_to_frame(data)

            # set the ACQUISITION_INFO to be our main dataframe
            datatable = waveform_data["ACQUISITION_INFO"]

            # leave out last 3 volumes (noise frames)
            datatable = datatable[datatable.VOLUME < num_frames]

            # loop through rows of ACQUISITION_INFO and compute the mean of the waveform value for each row
            for wave in waveform_data:
                # skip acquisition info
                if waveform_data[wave].attrs["LogDataType"] == "ACQUISITION_INFO":
                    continue

                waveform = waveform_data[wave]
                # rename VALUE column to wave name
                waveform.rename(columns={"VALUE": wave}, inplace=True)

                print(f"Waveform: {wave}")
                # get interval index
                interval_index = datatable.index
                # drop duplicates
                interval_index = interval_index.drop_duplicates()

                # get the indexer on the current waveform to line up the interval index ot the waveform index
                indexer = interval_index.get_indexer(waveform.ACQ_TIME_TIC)
                aligned_interval_index = interval_index.take(indexer)
                mask = indexer != -1
                waveform.set_index(aligned_interval_index, inplace=True)
                waveform = waveform[mask]
                waveform.drop(columns=["ACQ_TIME_TIC", "TIME"], inplace=True)

                # join the waveform data the aligned interval index
                datatable = datatable.join(waveform)

            # drop nan values
            datatable.dropna(inplace=True)

            # slice volume table
            slicetable = datatable.groupby(["VOLUME", "SLICE"]).mean()

            # group by volume and compute mean
            datatable = datatable.groupby("VOLUME").mean()

            # normalize (z-score) data
            for wave in waveform_data:
                if waveform_data[wave].attrs["LogDataType"] == "ACQUISITION_INFO":
                    continue
                slicetable[wave] = (slicetable[wave] - slicetable[wave].mean()) / slicetable[wave].std()
                # normalize the waveform data
                datatable[wave] = (datatable[wave] - datatable[wave].mean()) / datatable[wave].std()

            # load the field map data
            fmap_img = nib.load("fmap_aligned.nii.gz")
            fmap_volume = fmap_img.get_fdata()

            # construct a high pass filter and filter the fmap data
            tr = 1.761
            fs = 1 / tr
            fn = fs / 2
            w0_cutoff = 0.15 / fn
            b, a = iirfilter(2, w0_cutoff, btype="highpass", output="ba", ftype="butter")

            # filter fmap data
            print("Filtering field map data")
            fmap_volume_filtered = filtfilt(b, a, fmap_volume, axis=-1)

            # create a table for each slice
            fmap_volume_filtered_masked = np.ma.masked_array(
                fmap_volume_filtered,
                mask=np.broadcast_to(~brain_mask[..., np.newaxis], fmap_volume_filtered.shape),
            )
            fmap_slices_filtered = fmap_volume_filtered_masked.mean(axis=(0, 1)).data
            fmap_slices_filtered = (
                fmap_slices_filtered - fmap_slices_filtered.mean(axis=-1, keepdims=True)
            ) / fmap_slices_filtered.std(axis=-1, keepdims=True)

            # get resp by volume slice
            resp_volume = np.zeros(fmap_volume.shape[:-1])
            for i in range(fmap_volume_filtered.shape[2]):
                resp_volume[:, :, i] = np.corrcoef(
                    slicetable.loc[pd.IndexSlice[:, i], :]["RESP"], fmap_slices_filtered[i, :]
                )[0, 1]
            nib.Nifti1Image(resp_volume * brain_mask, fmap_img.affine).to_filename("resp_volume.nii.gz")
            # do same for pulse ox
            pulse_volume = np.zeros(fmap_volume.shape[:-1])
            for i in range(fmap_volume_filtered.shape[2]):
                pulse_volume[:, :, i] = np.corrcoef(
                    slicetable.loc[pd.IndexSlice[:, i], :]["PULS"], fmap_slices_filtered[i, :]
                )[0, 1]
            nib.Nifti1Image(pulse_volume * brain_mask, fmap_img.affine).to_filename("puls_volume.nii.gz")

            # average signal before filtering for comparison
            fmap_signal = fmap_volume[brain_mask, :].mean(axis=0)

            # average over the brain mask
            fmap_signal_filtered = fmap_volume_filtered[brain_mask, :].mean(axis=0)

            # normalize the fmap signal
            fmap_signal = (fmap_signal - fmap_signal.mean()) / fmap_signal.std()
            fmap_signal_filtered = (fmap_signal_filtered - fmap_signal_filtered.mean()) / fmap_signal_filtered.std()
            fmap_datatable = pd.DataFrame(
                {
                    "VOLUME": np.arange(fmap_signal.shape[0]),
                    "fmap_signal": fmap_signal,
                    "fmap_signal_filtered": fmap_signal_filtered,
                }
            )
            new_mask = brain_mask.copy()
            new_mask[..., 10:] = False
            fmap_signal2 = fmap_volume[new_mask, :].mean(axis=0)
            fmap_signal2 = (fmap_signal2 - fmap_signal2.mean()) / fmap_signal2.std()
            fmap_datatable2 = pd.DataFrame(
                {
                    "VOLUME": np.arange(fmap_signal.shape[0]),
                    "fmap_signal": fmap_signal2,
                }
            )

            # # puls
            # fig = plt.figure(figsize=(20, 5), layout="constrained")
            # subfigs = fig.subfigures(1, 2, wspace=0.1, hspace=0.1)
            # fig1 = subfigs[0]
            # f0, p0 = periodogram(datatable["PULS"], fs=fs)
            # f1, p1 = periodogram(fmap_signal2, fs=fs)
            # resp_power_spectra = pd.DataFrame({"Frequency (Hz)": f0, "Unfiltered": p0})
            # resp_power_spectra.set_index("Frequency (Hz)", inplace=True)
            # fmap_power_spectra = pd.DataFrame({"Frequency (Hz)": f1, "Unfiltered": p1})
            # fmap_power_spectra.set_index("Frequency (Hz)", inplace=True)
            # ax = fig1.subplots(2, 1)
            # sns.lineplot(data=resp_power_spectra, ax=ax[0])
            # sns.lineplot(data=fmap_power_spectra, ax=ax[1])
            # ax[0].set_ylim(0, 30)
            # ax[0].set_title("Power Spectrum of Puls Signal from Pulse Oximeter")
            # ax[1].set_ylim(0, 30)
            # fig2 = subfigs[1]
            # ax = fig2.subplots(2, 1)
            # sns.lineplot(data=datatable, x="VOLUME", y="PULS", ax=ax[0])
            # corr = np.corrcoef(datatable["PULS"], fmap_signal2)[0, 1]
            # ax[0].set_xlabel("Frame #")
            # ax[0].set_ylabel("Normalized Puls Signal")
            # ax[0].set_title("Puls Signal from Pulse Oximeter")
            # ax[0].set_ylim(-3, 3)
            # sns.lineplot(data=fmap_datatable2, x="VOLUME", y="fmap_signal", ax=ax[1])
            # ax[1].set_xlabel("Frame #")
            # ax[1].set_ylabel("Normalized Puls Signal")
            # ax[1].set_ylim(-3, 3)
            # plt.show()

            # plot power spectrum
            fig = plt.figure(figsize=(20, 5), layout="constrained")
            fig.suptitle(f"Run {run_idx + 1}")
            subfigs = fig.subfigures(1, 2, wspace=0.1, hspace=0.1)
            fig1 = subfigs[0]
            f0, p0 = periodogram(datatable["RESP"], fs=fs)
            f1, p1 = periodogram(fmap_signal, fs=fs)
            f2, p2 = periodogram(fmap_signal_filtered, fs=fs)
            resp_power_spectra = pd.DataFrame({"Frequency (Hz)": f0, "Unfiltered": p0})
            resp_power_spectra.set_index("Frequency (Hz)", inplace=True)
            fmap_power_spectra = pd.DataFrame({"Frequency (Hz)": f1, "Unfiltered": p1, "Filtered": p2})
            fmap_power_spectra.set_index("Frequency (Hz)", inplace=True)
            ax = fig1.subplots(2, 1)
            sns.lineplot(data=resp_power_spectra, ax=ax[0])
            sns.lineplot(data=fmap_power_spectra, ax=ax[1])
            ax[0].set_title("Power Spectrum of Respiration Signal from Respiratory Belt")
            ax[0].set_ylim(0, 60)
            ax[0].set_ylabel("Power Spectral Density")
            ax[1].set_title("Power Spectrum of Avg. MEDIC Field Map Signal")
            ax[1].axvline(w0_cutoff * fn, color="r", linestyle="--")
            ax[1].set_ylim(0, 60)
            ax[1].set_ylabel("Power Spectral Density")
            power_spectra = pd.DataFrame(
                {
                    "Frequency (Hz)": f0,
                    "resp_signal": p0,
                    "fmap_signal": p1,
                    "fmap_signal_filtered": p2,
                }
            )
            power_spectra.set_index("Frequency (Hz)", inplace=True)
            power_spectra.to_csv(DATA_DIR / f"power_spectra_run_{run_idx + 1:02d}.csv")

            # plot the curves
            pastel_colors = sns.color_palette("pastel")
            # fig2 = plt.figure(figsize=(10, 5), layout="constrained")
            fig2 = subfigs[1]
            ax = fig2.subplots(2, 1)
            sns.lineplot(data=datatable, x="VOLUME", y="RESP", ax=ax[0])
            corr = np.corrcoef(datatable["RESP"], fmap_signal_filtered)[0, 1]
            ax[0].set_xlabel("Frame #")
            ax[0].set_ylabel("Normalized Respiration Signal")
            ax[0].set_title("Respiration Signal from Respiratory Belt")
            ax[0].set_ylim(-3, 3)
            sns.lineplot(data=fmap_datatable, x="VOLUME", y="fmap_signal_filtered", ax=ax[1])
            ax[1].set_xlabel("Frame #")
            ax[1].set_ylabel("Normalized Respiration Signal")
            ax[1].set_title(f"Respiration Signal from MEDIC Field Map (R = {corr:.3f})")
            ax[1].set_ylim(-3, 3)
            fig.savefig(FIGURES_DIR / f"resp_run-{run_idx + 1}.png", dpi=300)
            resp_datatable = pd.DataFrame(
                {
                    "VOLUME": np.arange(datatable.shape[0]),
                    "resp_signal": datatable["RESP"].to_numpy(),
                    "fmap_signal": fmap_datatable["fmap_signal_filtered"].to_numpy(),
                }
            )
            resp_datatable.set_index("VOLUME", inplace=True)

            # Save resp data
            resp_datatable.to_csv(DATA_DIR / f"resp_data_run_{run_idx + 1:02d}.csv")

    plt.show()
    return 0
