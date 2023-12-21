import nibabel as nib
import numpy as np
from bids import BIDSLayout
from memori.pathman import PathManager as PathMan

from medic_analysis.common.figures import data_plotter, plt, render_dynamic_figure

from . import parser

# Define the path to the BIDS dataset
BIDS_DATA_DIR = "/home/usr/vana/GMT2/Andrew/SLICETEST"


def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))


def main():
    # add arguments to parser
    parser.add_argument(
        "--seed",
        nargs=3,
        type=int,
        help="Seed to use for correlation analysis.",
        default=[67, 33, 12],
    )
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

    # Load the dataset
    layout = BIDSLayout(args.bids_dir, database_path=args.bids_dir)

    # set output dir
    if args.output_dir is None:
        args.output_dir = (PathMan(args.bids_dir) / "derivatives").path

    # make sure the output dir exists
    if not PathMan(args.output_dir).exists():
        raise FileExistsError(f"Output directory {args.output_dir} does not exist.")

    # load the field map files
    runs = []
    for idx in range(3):
        run = idx + 1
        runs.append(nib.load(PathMan(args.output_dir) / f"run-{run:02d}" / "fmap.nii.gz"))

    if args.plot_only is None or 0 in args.plot_only:
        # get the seed to use for each image
        seed = args.seed

        # loop over runs
        corr_runs = []
        for img in runs:
            # get the data
            data = img.get_fdata()

            # get the seed
            seed_val = data[seed[0], seed[1], seed[2]][np.newaxis, :]

            # reshape data into 2D array
            data_mat = data.reshape(-1, data.shape[-1])

            # compute the correlation of the seed to all other voxels in image
            corr = corr2_coeff(seed_val, data_mat).reshape(data.shape[:-1])
            corr_runs.append(corr)

        # plot the correlation maps
        f0 = plt.figure(figsize=(8, 6), layout="constrained")
        data_plotter(
            corr_runs,
            figure=f0,
            colorbar=True,
            colorbar_label="Pearson Correlation",
            vmin=-1,
            vmax=1,
        )
        f0.text(0.6, 0.67, "(A) 72 Slices Interleaved", ha="center")
        f0.text(0.6, 0.35, "(B) 72 Slices Ascending", ha="center")
        f0.text(0.6, 0.02, "(C) 78 Slices Interleaved", ha="center")

    if args.plot_only is not None and 1 in args.plot_only:
        # make output directory for movie
        movies = PathMan(args.output_dir) / "movies"
        movies.mkdir(exist_ok=True)

        # runs = []
        # for idx in range(1):
        #     run = idx + 1
        #     runs.append(nib.load(PathMan(args.output_dir) / f"run-{run:02d}" / "fmap_native.nii.gz"))

        # loop over runs
        for idx, img in enumerate(runs):
            render_dynamic_figure(
                movies / f"run-{idx + 1:02d}.mp4",
                [img],
                colorbar=True,
                colorbar_alt_range=True,
            )

    # plot the maps
    plt.show()
