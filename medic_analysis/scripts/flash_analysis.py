from bids import BIDSLayout
import nibabel as nib
from memori.logging import run_process
from memori.pathman import PathManager as PathMan
from . import (
    parser,
)
from medic_analysis.common import data_plotter, plt, Figure

# Define the path to the BIDS dataset
BIDS_DATA_DIR = "/home/usr/vana/GMT2/Andrew/FLASHSUSTEST"


def main():
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

    # load up the appropriate files
    flash_fmap = PathMan(args.output_dir) / "flash1" / "romeo_output" / "B0.nii"
    flash_mask = PathMan(args.output_dir) / "flash1" / "brain_mask.nii.gz"
    topup_fmap = PathMan(args.output_dir) / "topup" / "fout_hres.nii.gz"
    medic_fmap = PathMan(args.output_dir) / "medic" / "fmap_hres.nii.gz"
    flash_fmap_img = nib.load(flash_fmap.path)
    flash_mask_img = nib.load(flash_mask.path)
    topup_fmap_img = nib.load(topup_fmap.path)
    medic_fmap_img = nib.load(medic_fmap.path)

    # now mask the data
    flash_fmap_data = flash_fmap_img.get_fdata()
    topup_fmap_data = topup_fmap_img.get_fdata()[..., 0]
    medic_fmap_data = medic_fmap_img.get_fdata()[..., 0]

    # make figure
    f = plt.figure(figsize=(16, 8), layout="constrained")
    sf = f.subfigures(1, 2)

    # plot field maps
    data_plotter(
        [topup_fmap_data, flash_fmap_data, medic_fmap_data],
        colormaps="gray",
        slices=(125, 165, 65),
        colorbar=True,
        colorbar_alt_range=True,
        figure=sf[0],
    )
    # figure label text
    sf[0].text(0.6, 0.67, "(A) PE-Polar (topup)", ha="center")
    sf[0].text(0.6, 0.35, "(B) FLASH (ROMEO)", ha="center")
    sf[0].text(0.6, 0.02, "(C) ME-EPI (MEDIC)", ha="center")

    # compute difference from flash
    topup_diff = (flash_fmap_data - topup_fmap_data)
    medic_diff = (flash_fmap_data - medic_fmap_data)
    data_plotter(
        [topup_diff, medic_diff],
        slices=(125, 165, 65),
        vmin=-100,
        vmax=100,
        colorbar2=True,
        colorbar2_alt_range=True,
        figure=sf[1],
    )
    # figure label text
    sf[1].text(0.4, 0.59, "(D) FLASH (ROMEO) - PE-Polar (topup)", ha="center")
    sf[1].text(0.4, 0.09, "(E) FLASH (ROMEO) - ME-EPI (MEDIC)", ha="center")

    # plot figures
    plt.show()
