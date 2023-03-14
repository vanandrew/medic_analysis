#!/usr/bin/env python
import warnings
import sys
from tempfile import TemporaryDirectory
from memori.pathman import PathManager as Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.image import AxesImage
import seaborn as sns
import nibabel as nib
import numpy as np
import numpy.typing as npt
from typing import Callable, cast, List, Tuple, Sequence, Union
from memori.logging import run_process

# plot settings
warnings.filterwarnings("ignore")
sns.set(font="Lato", style="dark")
plt.style.use("dark_background")

# color palettes
dark_div_1 = "icefire"


def normalize(x: npt.NDArray) -> npt.NDArray:
    return (x - x.min()) / (x.max() - x.min())


def ffmpeg(img_dir: str, out_file: str, fps: int = 20) -> None:
    run_process(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(Path(img_dir) / "temp_%05d.png"),
            "-c:v",
            "libx264",
            "-crf",
            "25",
            "-pix_fmt",
            "yuv420p",
            out_file,
        ]
    )


def hz_limits_to_mm(
    hz_min: float = -100, hz_max: float = 200, total_readout_time: float = 0.0305196, resolution: float = 2
) -> Tuple[float, float]:
    d_min = hz_min * total_readout_time * resolution
    d_max = hz_max * total_readout_time * resolution
    return d_min, d_max


def subplot_imshow(
    f: Union[Figure, SubFigure], data: npt.NDArray, entry: Tuple[int, int, int], vmin: float, vmax: float, cmap: str
) -> Tuple[Axes, AxesImage]:
    ax = f.add_subplot(*entry, frame_on=False, anchor="C")
    im = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return ax, im


def data_plotter(
    imgs: Union[List[npt.NDArray], List[nib.Nifti1Image]],
    colormaps: Union[List[str], str] = "icefire",
    slices: Tuple[int, int, int] = (54, 54, 30),
    vmin: float = -100,
    vmax: float = 200,
    ishz: bool = False,
    colorbar: bool = False,
    colorbar_label: str = "Hz",
    colorbar_source_idx: Tuple[int, int] = (0, 0),
    colorbar_alt_range: bool = False,
    colorbar_alt_range_fx: Callable = hz_limits_to_mm,
    colorbar_alt_label: str = "mm",
    colorbar_aspect: int = 60,
    colorbar2: bool = False,
    colorbar2_label: str = "Hz",
    colorbar2_source_idx: Tuple[int, int] = (0, 0),
    colorbar2_alt_range: bool = False,
    colorbar2_alt_range_fx: Callable = hz_limits_to_mm,
    colorbar2_alt_label: str = "mm",
    colorbar2_aspect: int = 60,
    figsize: Sequence[float] = (8, 9),
    figure: Union[Figure, SubFigure, None] = None,
) -> Union[Figure, SubFigure]:
    # if imgs has nib.Nifti1Image type in list, get numpy array data
    if type(imgs[0]) is nib.Nifti1Image:
        imgs = cast(List[nib.Nifti1Image], imgs)
        imgs = [i.get_fdata() for i in imgs]
    imgs = cast(List[npt.NDArray], imgs)

    # extend color maps into list if a string
    if type(colormaps) is str:
        colormaps = [colormaps] * len(imgs)
    colormaps = cast(List[str], colormaps)

    # check if existing figure passed in
    # if it is, use it
    # if not, just create a new figure
    if figure is not None:
        f = figure
    else:
        f = plt.figure(figsize=figsize, layout="constrained")

    # get grid size based on input
    grid_size = (len(imgs), 3)

    # plot each img on a row
    plot_idx = 1
    fig_row = []
    for cmap, img in zip(colormaps, imgs):
        ax1, axi1 = subplot_imshow(f, img[:, :, slices[2]].T, (*grid_size, plot_idx), vmin, vmax, cmap)
        plot_idx += 1
        ax2, axi2 = subplot_imshow(f, img[slices[0], :, :].T, (*grid_size, plot_idx), vmin, vmax, cmap)
        plot_idx += 1
        ax3, axi3 = subplot_imshow(f, img[:, slices[1], :].T, (*grid_size, plot_idx), vmin, vmax, cmap)
        plot_idx += 1
        fig_row.append(((ax1, axi1), (ax2, axi2), (ax3, axi3)))

    # if colorbar is set, draw it
    if colorbar:
        colorbar_source_img = fig_row[colorbar_source_idx[0]][colorbar_source_idx[1]][1]
        cbar = f.colorbar(
            colorbar_source_img,
            ax=[r[0][0] for r in fig_row],
            aspect=colorbar_aspect,
            pad=0.35,
            location="left",
            orientation="vertical",
        )
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.set_ylabel(colorbar_label, labelpad=-10, rotation=90)
        # for colorbar alt range
        if colorbar_alt_range:
            alt_vmin, alt_vmax = colorbar_alt_range_fx(vmin, vmax)
            cax = cbar.ax.twinx()
            cax.set_ylim(alt_vmin, alt_vmax)
            cax.set_ylabel(colorbar_alt_label, labelpad=-5, rotation=90)

    # if colorbar2 is set, draw it
    if colorbar2:
        colorbar2_source_img = fig_row[colorbar2_source_idx[0]][colorbar2_source_idx[1]][1]
        cbar = f.colorbar(
            colorbar2_source_img,
            ax=[r[-1][0] for r in fig_row],
            aspect=colorbar2_aspect,
            pad=0.35,
            location="right",
            orientation="vertical",
        )
        cbar.ax.yaxis.set_ticks_position("right")
        cbar.ax.set_ylabel(colorbar2_label, labelpad=-5, rotation=90)
        # for colorbar alt range
        if colorbar2_alt_range:
            alt_vmin, alt_vmax = colorbar2_alt_range_fx(vmin, vmax)
            cax = cbar.ax.twinx()
            cax.yaxis.set_ticks_position("left")
            cax.set_ylim(alt_vmin, alt_vmax)
            cax.set_ylabel(colorbar2_alt_label, labelpad=-50, rotation=90)
            cbar.ax.yaxis.set_ticks_position("right")  # reset the ticks position on the non-alt bar

    # return figure
    return f


# # render dynamic figure
# def render_dynamic_figure(
#     out_file, img, plot_func=plot_single, extra_title=None, img_labels=None, n_frames=None, **kwargs
# ):
#     if "img1_label" in kwargs:
#         img_label = kwargs["img1_label"]
#     else:
#         img_label = ""
#     # create temporary directory
#     with TemporaryDirectory() as temp_dir:
#         if n_frames is None:
#             n_frames = img.shape[-1]
#         for i in range(n_frames):
#             print(f"frame: {i}")
#             if img_labels:
#                 img_label = img_labels[i]
#             else:
#                 img_label = None

#             # get data
#             if type(img) == nib.nifti1.Nifti1Image:
#                 img_data = img.dataobj[..., i]
#             else:
#                 img_data = img[..., i]

#             # extra titles
#             if extra_title is not None:
#                 ex_title = extra_title[i]
#             else:
#                 ex_title = None

#             # plot the image
#             f = plot_func(img_data, img_label=img_label, extra_title=ex_title, **kwargs)

#             # save the figure
#             f.savefig(str(Path(temp_dir) / f"temp_{i:05d}.png"))

#             # close the figure
#             plt.close(f)

#         # run ffmpeg
#         ffmpeg(temp_dir, out_file)


# def alignment_check(out_file, runs, ref, run_labels=None, n_frames=100, **kwargs):
#     # create temporary directory
#     with TemporaryDirectory() as temp_dir:
#         k = 0
#         for run_idx, r in enumerate(runs):
#             for i in range(n_frames):
#                 if run_labels:
#                     kwargs["img1_label"] = f"{run_labels[run_idx]}\nframe: {i}"
#                 print(f"frame: {k}")

#                 # if only single frame plot single
#                 if n_frames == 1:
#                     data = r.get_fdata()
#                 else:
#                     data = r.dataobj[..., i]

#                 # plot the image
#                 f = plot_double(data, ref, colorbar=False, **kwargs)

#                 # save the figure
#                 f.savefig(str(Path(temp_dir) / f"temp_{k:05d}.png"))

#                 # close the figure
#                 plt.close(f)

#                 # increment counter
#                 k += 1

#         # run ffmpeg
#         ffmpeg(temp_dir, out_file)


# def alignment_check_single(out_file, runs, run_labels=None, n_frames=100, extra=None, **kwargs):
#     # create temporary directory
#     with TemporaryDirectory() as temp_dir:
#         k = 0
#         for run_idx, r in enumerate(runs):
#             for i in range(n_frames):
#                 if run_labels:
#                     kwargs["img_label"] = f"{run_labels[run_idx]}\nframe: {i}"
#                 print(f"frame: {k}")

#                 # if only single frame plot single
#                 if n_frames == 1:
#                     data = r.get_fdata()
#                 else:
#                     data = r.dataobj[..., i]

#                 # plot the image
#                 f = plot_single(data, colorbar=False, **kwargs)

#                 # add extra plot
#                 if extra:
#                     extra()

#                 # save the figure
#                 f.savefig(str(Path(temp_dir) / f"temp_{k:05d}.png"))

#                 # close the figure
#                 plt.close(f)

#                 # increment counter
#                 k += 1

#         # run ffmpeg
#         ffmpeg(temp_dir, out_file)
