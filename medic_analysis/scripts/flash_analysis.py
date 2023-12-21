"""Compare FLASH, TOPUP, and MEDIC field maps.

This analysis is not included in the preprint.
"""
import nibabel as nib
from memori.pathman import PathManager as PathMan

from medic_analysis.common.figures import FIGURE_OUT, data_plotter, plt

from . import parser

# Define the path to the BIDS dataset
BIDS_DATA_DIR = "/home/usr/vana/GMT2/Andrew/FLASHSUSTEST"


def main():
    # call the parser
    args = parser.parse_args()

    # if bids dir not specified, use default
    if args.bids_dir is None:
        args.bids_dir = BIDS_DATA_DIR

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
    flash_fmap_data = flash_fmap_img.get_fdata() * flash_mask_img.get_fdata()
    topup_fmap_data = topup_fmap_img.get_fdata()[..., 0] * flash_mask_img.get_fdata()
    medic_fmap_data = medic_fmap_img.get_fdata()[..., 0] * flash_mask_img.get_fdata()

    # make figure
    f = plt.figure(figsize=(16, 8), layout="constrained")
    sf = f.subfigures(1, 2)

    crop = (slice(35, 225), slice(32, 222), slice(0, 176))
    slices = (95, 100, 65)

    # plot field maps
    data_plotter(
        [topup_fmap_data[crop], flash_fmap_data[crop], medic_fmap_data[crop]],
        colormaps="gray",
        slices=slices,
        colorbar=True,
        colorbar_alt_range=True,
        figure=sf[0],
        text_color="white",
    )
    sbs = sf[0].get_axes()
    sbs[1].set_title("(A) PE-Polar (topup)", loc="center", y=-0.25)
    sbs[4].set_title("(B) FLASH (ROMEO)", loc="center", y=-0.25)
    sbs[7].set_title("(C) ME-EPI (MEDIC)", loc="center", y=-0.25)

    # compute difference from flash
    fmap_diff = medic_fmap_data - topup_fmap_data
    topup_diff = topup_fmap_data - flash_fmap_data
    medic_diff = medic_fmap_data - flash_fmap_data
    data_plotter(
        [fmap_diff[crop], topup_diff[crop], medic_diff[crop]],
        slices=slices,
        vmin=-50,
        vmax=50,
        colorbar2=True,
        colorbar2_alt_range=True,
        figure=sf[1],
        text_color="white",
    )
    sbs = sf[1].get_axes()
    sbs[1].set_title("(D) ME-EPI (MEDIC) - PE-Polar (topup)", loc="center", y=-0.25)
    sbs[4].set_title("(E) PE-Polar (topup) - FLASH (ROMEO)", loc="center", y=-0.25)
    sbs[7].set_title("(F) ME-EPI (MEDIC) - FLASH (ROMEO)", loc="center", y=-0.25)

    # plot figures
    f.savefig(FIGURE_OUT / "fieldmap_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
