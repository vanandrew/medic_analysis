import sys
import logging
from typing import Dict, List
from memori.pathman import PathManager as PathMan
from memori.helpers import working_directory
from bids import BIDSLayout
import numpy as np
import pandas as pd
import nibabel as nib
import pydicom
from warpkit.distortion import medic
from warpkit.unwrap import create_brain_mask
from warpkit.utilities import displacement_map_to_field, resample_image
from medic_analysis.common import apply_framewise_mats, framewise_align, sns, plt, run_topup
import simplebrainviewer as sbv
from . import (
    parser,
    PED_TABLE,
    POLARITY_IDX,
)


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
        frame.set_index("ACQ_TIME_TIC", inplace=True)

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

    # return the dataframe
    return frame


def main():
    # add to parser
    parser.add_argument("--runs", nargs="+", default=["02"])
    parser.add_argument("--num_frames", type=int, default=510, help="number of frames in the scan")

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

    # loop over runs for processing
    for run in args.runs:
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

            # compute motion parameters
            framewise_align("ref.nii", mag[0].path, "mcflirt")

            # load in motion params, convert rotations to mm
            motion_params = np.loadtxt("mcflirt.par")
            motion_params[:, :3] = np.rad2deg(motion_params[:, :3])
            motion_params[:, :3] = 50 * (np.pi / 180) * motion_params[:, :3]
            motion_params = motion_params[:args.num_frames]

            # run medic
            # fmap_native, dmap, fmap = medic(
            #     phase_imgs,
            #     mag_imgs,
            #     TEs,
            #     total_readout_time,
            #     phase_encoding_direction,
            #     n_cpus=8,
            #     frames=list(range(args.num_frames)),
            #     border_size=5,
            #     svd_filt=30,
            # )
            # fmap_native.to_filename("fmap_native.nii.gz")
            # dmap.to_filename("dmap.nii.gz")
            # fmap.to_filename("fmap.nii.gz")

            # make brain mask of reference image
            ref_img = nib.load("ref.nii")
            mask = create_brain_mask(ref_img.get_fdata(), 0)

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
            datatable = datatable[datatable.VOLUME < args.num_frames]

            # split data by slice
            slice_tables = dict()
            for slice_num in datatable.SLICE.unique():
                slice_tables[slice_num] = datatable[datatable.SLICE == slice_num]

            volume_tables = dict()
            # loop through rows of ACQUISITION_INFO and compute the mean of the waveform value for each row
            for wave in waveform_data:
                # skip acquisition info
                if waveform_data[wave].attrs["LogDataType"] == "ACQUISITION_INFO":
                    continue

                # loop through each slice in a single band (for multi-band 6 this is slices 0 - 11)
                for slice_num in range(12):
                    # get the slice table
                    slice_table = slice_tables[slice_num]

                    # get the indexer for the slice table and apply it to the waveform data
                    waveform_data[wave]["interval"] = slice_table.index.get_indexer(waveform_data[wave].index)

                    # for the subset, rename VALUE to the wave name
                    waveform_data[wave] = waveform_data[wave].rename(columns={"VALUE": wave})

                    # group the subset by the interval and take mean
                    mean_values = waveform_data[wave].groupby("interval").mean()
                    # remove the -1 intervals
                    mean_values = mean_values[mean_values.index != -1]

                    # get size of slice_table
                    slice_table_size = slice_table.shape[0]

                    # if size of mean_values is not equal to slice_table_size, figure out which indices are missing
                    # and fill them in with the mean of the present indices
                    if mean_values.shape[0] != slice_table_size:
                        # get the indices of the slice table
                        slice_table_indices = np.arange(slice_table_size)

                        # get the indices of the mean values
                        mean_values_indices = mean_values.index.values

                        # get the indices that are missing
                        missing_indices = np.setdiff1d(slice_table_indices, mean_values_indices)

                        # get the internal for each missing index
                        missing_intervals = slice_table.iloc[missing_indices].index

                        # for each missing interval get the closest value in the waveform data for start and finish
                        # average them to get the substitute value
                        for inter, indi in zip(missing_intervals, missing_indices):
                            # get indices closest to start and end of interval
                            start_min = np.argmin(np.abs(inter.left - waveform_data[wave].index))
                            end_min = np.argmin(np.abs(inter.right - waveform_data[wave].index))

                            # get the mean of the start and end values
                            mean_values.loc[indi] = waveform_data[wave].iloc[[start_min, end_min]].mean()

                    # merge into slice_table
                    slice_table.set_index(mean_values.index, inplace=True)
                    slice_table = slice_table.merge(mean_values, left_index=True, right_index=True)

                    # reset the interval index for the slice table
                    slice_table.set_index(
                        pd.IntervalIndex.from_arrays(
                            slice_table.ACQ_START_TICS, slice_table.ACQ_FINISH_TICS, closed="left"
                        ),
                        inplace=True,
                    )

                    # now summarize over the VOLUME for the slice, this will average over echoes
                    volume_tables[slice_num] = slice_table.groupby("VOLUME").mean()

                    # store the slice table
                    slice_tables[slice_num] = slice_table

            # # draw plots
            # f = plt.figure(figsize=(16, 8), layout="constrained")
            # subplots = f.subplots(2, 1)
            # for slice_num in range(12):
            #     sns.lineplot(
            #         data=volume_tables[slice_num], x="VOLUME", y="RESP", label=f"Slice {slice_num}", ax=subplots[0]
            #     )
            #     sns.lineplot(
            #         data=volume_tables[slice_num], x="VOLUME", y="PULS", label=f"Slice {slice_num}", ax=subplots[1]
            #     )
            # subplots[0].set_title("RESP")
            # subplots[1].set_title("PULS")

            # load the field map data
            fmap_img = nib.load("fmap.nii.gz")

            # build the respiration data into an entire volume
            resp_volume = np.zeros(fmap_img.shape)
            for slice_idx in range(fmap_img.shape[2]):
                # assign the resp data to the volume
                resp_volume[:, :, slice_idx, :] = volume_tables[slice_idx % 12]["RESP"].values[
                    np.newaxis, np.newaxis, :
                ]

            # for each slice compute regression model of resp vs. fmap
            fmap_data = fmap_img.get_fdata()
            residual_map = np.zeros(fmap_img.shape[:-1])
            variance_map = np.zeros(fmap_img.shape[:-1])
            residual_map2 = np.zeros(fmap_img.shape[:-1])
            variance_map2 = np.zeros(fmap_img.shape[:-1])
            r2_map = np.zeros(fmap_img.shape[:-1])
            for slice_idx in range(fmap_img.shape[2]):
                # first mask the current slice with the brain mask
                fmap_slice = fmap_data[mask[..., slice_idx], slice_idx, :]
                resp_trace = volume_tables[slice_idx % 12]["RESP"].values

                # build the design matrix for this slice
                design_matrix = np.stack((np.ones(resp_trace.shape), resp_trace), axis=1)
                # add motion params
                design_matrix = np.concatenate((design_matrix, motion_params), axis=1)

                # build response matrix for this slice
                response_matrix = fmap_slice.T

                # compute the regression on this model
                _, residuals, _, _ = np.linalg.lstsq(design_matrix, response_matrix, rcond=None)

                # store the residuals in the residual map
                residual_map[mask[..., slice_idx], slice_idx] = residuals

                # compute variance and store in variance map
                variance_map[mask[..., slice_idx], slice_idx] = fmap_slice.var(axis=1) * fmap_slice.shape[1]

                # just motion params
                design_matrix2 = np.concatenate((np.ones(resp_trace.shape)[:, np.newaxis], motion_params), axis=1)
                _, residuals2, _, _ = np.linalg.lstsq(design_matrix2, response_matrix, rcond=None)
                residual_map2[mask[..., slice_idx], slice_idx] = residuals2
                variance_map2[mask[..., slice_idx], slice_idx] = fmap_slice.var(axis=1) * fmap_slice.shape[1]

            # compute the r2 map
            r2_map = 1 - np.divide(
                residual_map, variance_map, out=np.zeros_like(residual_map), where=(variance_map != 0)
            )

            # compute the r2 map for just motion params
            r2_map2 = 1 - np.divide(
                residual_map2, variance_map2, out=np.zeros_like(residual_map2), where=(variance_map2 != 0)
            )

            # set areas outside map to 0
            r2_map[~mask] = 0
            r2_map2[~mask] = 0

            # plot the r2 map
            # sbv.plot_brain(r2_map)
            diff = r2_map - r2_map2

            # eta2_avg = diff.sum(axis=0).sum(axis=0) / mask.sum(axis=0).sum(axis=0)
            # num_voxels_slice = mask.sum(axis=0).sum(axis=0)
            # plt.figure()
            # sns.scatterplot(data={"slice_times": slice_times, "eta2_avg": eta2_avg}, x="slice_times", y="eta2_avg")
            # plt.figure()
            # plt.plot(num_voxels_slice)
            # plt.show()

            # save the r2 map to file
            nib.Nifti1Image(r2_map, fmap_img.affine).to_filename("r2_map.nii.gz")
            nib.Nifti1Image(r2_map2, fmap_img.affine).to_filename("r2_map_motion_only.nii.gz")
            nib.Nifti1Image(diff, fmap_img.affine).to_filename("r2_map_respiration.nii.gz")

            # apply corrections to the data using the field map
            # load the medic displacement maps
            displacement_maps = "dmap.nii.gz"
            dmaps_img = nib.load(displacement_maps)

            # get the first echo images to correct
            first_echo_img = mag_imgs[0]

            # for each frame apply the correction
            corrected_data = np.zeros((*first_echo_img.shape[:3], args.num_frames))
            for frame_idx in range(args.num_frames):
                logging.info(f"Correcting Frame: {frame_idx}")
                # get the frame to correct
                frame_data = first_echo_img.dataobj[..., frame_idx]

                # make into image
                frame_img = nib.Nifti1Image(frame_data, first_echo_img.affine)

                # get the dmap for this frame
                dmap_data = dmaps_img.dataobj[..., frame_idx]

                # make into image
                dmap_img = nib.Nifti1Image(dmap_data, dmaps_img.affine)

                # get displacement field
                dfield_img = displacement_map_to_field(dmap_img)

                # apply the correction
                corrected_img = resample_image(ref_img, frame_img, dfield_img)

                # store the corrected_frame
                corrected_data[..., frame_idx] = corrected_img.get_fdata()

            # save the corrected data
            corrected_img = nib.Nifti1Image(corrected_data, first_echo_img.affine)
            corrected_img.to_filename("medic_corrected.nii.gz")

            # get new ref img
            new_ref_img = nib.Nifti1Image(corrected_data[..., 100], first_echo_img.affine)
            new_ref_img.to_filename("new_ref.nii")

            # compute motion parameters
            framewise_align("new_ref.nii", "medic_corrected", "mcflirt_corrected")

            # apply moco params from distorted align
            apply_framewise_mats("new_ref.nii", "medic_corrected", "mcflirt.mat", "mcflirt_dcorrected")
