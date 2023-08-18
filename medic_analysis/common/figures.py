#!/usr/bin/env python
import warnings
import logging
from tempfile import TemporaryDirectory
from memori.pathman import PathManager as Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.image import AxesImage
import seaborn as sns
import nibabel as nib
import numpy.typing as npt
from typing import Callable, cast, List, Tuple, Optional, Sequence, Union
from memori.logging import run_process

# plot settings
warnings.filterwarnings("ignore")
sns.set(
    font="Lato",
    font_scale=1,
    palette="pastel",
    style="white",
    # style="dark",
    # rc={
    #     "axes.facecolor": "black",
    #     "figure.facecolor": "black",
    #     "axes.labelcolor": "white",
    #     "axes.titlecolor": "white",
    #     "text.color": "white",
    #     "xtick.color": "white",
    #     "ytick.color": "white",
    # },
)

# figure output directory
FIGURE_OUT = Path("/home/usr/vana/GMT2/Andrew/medic_analysis/figures")
FIGURE_OUT.mkdir(exist_ok=True, parents=True)


def normalize(x: npt.NDArray) -> npt.NDArray:
    """Normalize an array to the range [0, 1].

    Parameters
    ----------
    x : npt.NDArray
        Data to normalize.

    Returns
    -------
    npt.NDArray
        Normalized data.
    """
    return (x - x.min()) / (x.max() - x.min())


def ffmpeg(img_dir: Union[str, Path], out_file: Union[str, Path], fps: int = 20) -> None:
    """Create a video from a directory of images.

    Parameters
    ----------
    img_dir : Union[str, Path]
        Directory containing images.
    out_file : Union[str, Path]
        Output video file.
    fps : int, optional
        Frames per second, by default 20

    Returns
    -------
    None
    """
    if type(img_dir) is Path:
        img_dir = img_dir.path
    img_dir = cast(str, img_dir)
    if type(out_file) is Path:
        out_file = out_file.path
    out_file = cast(str, out_file)
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
    """Convert Hz limits to mm limits.

    Parameters
    ----------
    hz_min : float, optional
        Minimum Hz value, by default -100
    hz_max : float, optional
        Maximum Hz value, by default 200
    total_readout_time : float, optional
        Total readout time, by default 0.0305196
    resolution : float, optional
        Resolution, by default 2

    Returns
    -------
    Tuple[float, float]
        Minimum and maximum mm values.
    """
    d_min = hz_min * total_readout_time * resolution
    d_max = hz_max * total_readout_time * resolution
    return d_min, d_max


def subplot_imshow(
    f: Union[Figure, SubFigure],
    data: npt.NDArray,
    entry: Tuple[int, int, int],
    vmin: float,
    vmax: float,
    cmap: str,
    axes_list: Optional[List[Axes]] = None,
) -> Tuple[Axes, AxesImage]:
    """Create a subplot with an image.

    Parameters
    ----------
    f : Union[Figure, SubFigure]
        Figure to add subplot to.
    data : npt.NDArray
        Data to plot.
    entry : Tuple[int, int, int]
        Entry for subplot.
    vmin : float
        Minimum value for colorbar.
    vmax : float
        Maximum value for colorbar.
    cmap : str
        Colormap to use.
    axes_list : Optional[List[Axes]], optional
        List of axes to add to instead of creating a new axes, by default None

    Returns
    -------
    Tuple[Axes, AxesImage]
        Axes and image.
    """
    if axes_list is not None:
        plot_idx = entry[-1] - 1
        ax = axes_list[plot_idx]
    else:
        ax = f.add_subplot(*entry, frame_on=False, anchor="C")  # type: ignore
    im = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return ax, im


def data_plotter(
    imgs: Union[List[npt.NDArray], List[nib.Nifti1Image]],
    colormaps: Union[List[str], str] = "icefire",
    slices: Tuple[int, int, int] = (55, 55, 36),
    vmin: Union[List[float], float] = -100,
    vmax: Union[List[float], float] = 150,
    colorbar: bool = False,
    colorbar_label: str = "Hz",
    colorbar_labelpad: int = -5,
    colorbar_source_idx: Tuple[int, int] = (0, 0),
    colorbar_alt_range: bool = False,
    colorbar_alt_labelpad: int = -5,
    colorbar_alt_range_fx: Callable = hz_limits_to_mm,
    colorbar_alt_label: str = "mm",
    colorbar_aspect: int = 60,
    colorbar_pad: float = 0.1,
    colorbar2: bool = False,
    colorbar2_label: str = "Hz",
    colorbar2_labelpad: int = -5,
    colorbar2_source_idx: Tuple[int, int] = (0, 0),
    colorbar2_alt_range: bool = False,
    colorbar2_alt_labelpad: int = -50,
    colorbar2_alt_range_fx: Callable = hz_limits_to_mm,
    colorbar2_alt_label: str = "mm",
    colorbar2_aspect: int = 60,
    colorbar2_pad: float = 0.1,
    figsize: Sequence[float] = (5, 6),
    figure: Union[Figure, SubFigure, None] = None,
    frame_num: int = 0,
    text_color: str = "black",
    fontsize: int = 12,
    grid_size: Optional[Tuple[int, int]] = None,
    cbar_ax: Optional[Axes] = None,
    cbar2_ax: Optional[Axes] = None,
    axes_list: Optional[List[Axes]] = None,
    fraction: float = 0.15,
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

    # extend vmin into list if a float
    if type(vmin) is not list:
        vmin = cast(float, vmin)
        vmin = [vmin] * len(imgs)
    vmin = cast(List[float], vmin)

    # extend vmax into list if a float
    if type(vmax) is not list:
        vmax = cast(float, vmax)
        vmax = [vmax] * len(imgs)
    vmax = cast(List[float], vmax)

    # check if existing figure passed in
    # if it is, use it
    # if not, just create a new figure
    if figure is not None:
        f = figure
    else:
        f = plt.figure(figsize=figsize, layout="constrained")

    # get grid size based on input
    if grid_size is None:
        grid_size = (len(imgs), 3)

    # plot each img on a row
    plot_idx = 1
    fig_row = []
    for cmap, vmin_i, vmax_i, img in zip(colormaps, vmin, vmax, imgs):
        if len(img.shape) == 4:
            img = img[:, :, :, frame_num]
        ax1, axi1 = subplot_imshow(f, img[:, :, slices[2]].T, (*grid_size, plot_idx), vmin_i, vmax_i, cmap, axes_list)
        plot_idx += 1
        ax2, axi2 = subplot_imshow(f, img[slices[0], :, :].T, (*grid_size, plot_idx), vmin_i, vmax_i, cmap, axes_list)
        plot_idx += 1
        ax3, axi3 = subplot_imshow(f, img[:, slices[1], :].T, (*grid_size, plot_idx), vmin_i, vmax_i, cmap, axes_list)
        plot_idx += 1
        fig_row.append(((ax1, axi1), (ax2, axi2), (ax3, axi3)))

    # if colorbar is set, draw it
    if colorbar:
        colorbar_source_img = fig_row[colorbar_source_idx[0]][colorbar_source_idx[1]][1]
        cbar = f.colorbar(
            colorbar_source_img,
            ax=cbar_ax if cbar_ax is not None else [r[0][0] for r in fig_row],
            aspect=colorbar_aspect,
            fraction=fraction,
            pad=colorbar_pad,
            location="left",
            orientation="vertical",
        )
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.set_ylabel(colorbar_label, labelpad=colorbar_labelpad, color=text_color, fontsize=fontsize, rotation=90)
        cbar.ax.tick_params(color=text_color, labelsize=fontsize, labelcolor=text_color)
        # for colorbar alt range
        if colorbar_alt_range:
            alt_vmin, alt_vmax = colorbar_alt_range_fx(vmin[colorbar_source_idx[0]], vmax[colorbar_source_idx[0]])
            cax = cbar.ax.twinx()
            cax.set_ylim(alt_vmin, alt_vmax)
            cax.set_ylabel(
                colorbar_alt_label, labelpad=colorbar_alt_labelpad, color=text_color, fontsize=fontsize, rotation=90
            )
            cax.tick_params(color=text_color, labelsize=fontsize, labelcolor=text_color)

    # if colorbar2 is set, draw it
    if colorbar2:
        colorbar2_source_img = fig_row[colorbar2_source_idx[0]][colorbar2_source_idx[1]][1]
        cbar = f.colorbar(
            colorbar2_source_img,
            ax=cbar2_ax if cbar2_ax is not None else [r[-1][0] for r in fig_row],
            aspect=colorbar2_aspect,
            fraction=fraction,
            pad=colorbar2_pad,
            location="right",
            orientation="vertical",
        )
        cbar.ax.yaxis.set_ticks_position("right")
        cbar.ax.set_ylabel(
            colorbar2_label, labelpad=colorbar2_labelpad, color=text_color, fontsize=fontsize, rotation=90
        )
        cbar.ax.tick_params(color=text_color, labelsize=fontsize, labelcolor=text_color)
        # for colorbar alt range
        if colorbar2_alt_range:
            alt_vmin, alt_vmax = colorbar2_alt_range_fx(vmin[colorbar2_source_idx[0]], vmax[colorbar2_source_idx[0]])
            cax = cbar.ax.twinx()
            cax.yaxis.set_ticks_position("left")
            cax.set_ylim(alt_vmin, alt_vmax)
            cax.set_ylabel(
                colorbar2_alt_label, labelpad=colorbar2_alt_labelpad, color=text_color, fontsize=fontsize, rotation=90
            )
            cax.tick_params(color=text_color, labelsize=fontsize, labelcolor=text_color)
            cbar.ax.yaxis.set_ticks_position("right")  # reset the ticks position on the non-alt bar

    # return figure
    return f


# render dynamic figure
def render_dynamic_figure(
    out_file: Union[str, Path],
    imgs: Union[List[npt.NDArray], List[nib.Nifti1Image]],
    figure_fx: Union[Callable, None] = None,
    fps: int = 10,
    **kwargs,
):
    # create temporary directory
    with TemporaryDirectory() as temp_dir:
        # make sure images 4D
        if not all([len(i.shape) == 4 for i in imgs]):
            raise ValueError("All images must be 4D")

        # make sure number of frames is the same for all imgs
        if not all([i.shape[3] == imgs[0].shape[3] for i in imgs]):
            raise ValueError("All images must have the same number of frames")

        # get number of frames
        num_frames = imgs[0].shape[3]

        # loop through frames
        for frame_num in range(num_frames):
            logging.info(f"Rendering frame {frame_num+1} of {num_frames}")
            # plot data and get figure
            figure = data_plotter(imgs=imgs, **kwargs, frame_num=frame_num)

            # modify the figure with figure_fx if it exists
            if figure_fx is not None:
                figure = figure_fx(figure, frame_num)

            # save figure to file in temp dir
            figure = cast(Figure, figure)
            figure.savefig(str(Path(f"{temp_dir}") / f"temp_{frame_num:05d}.png"))

            # close the figure
            plt.close(figure)

        # run ffmpeg
        ffmpeg(temp_dir, out_file, fps)
