from bids import BIDSLayout
import nibabel as nib
from memori.pathman import PathManager as PathMan
from . import (
    parser,
)

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
    topup_fmap = PathMan(args.output_dir) / "topup" / "fout.nii.gz"
    medic_fmap = PathMan(args.output_dir) / "medic_ms" / "fmap.nii.gz"
    func_mask = PathMan(args.output_dir) / "func" / "brain_mask_ref.nii.gz"
    flash_fmap_img = nib.load(flash_fmap.path)
    flash_mask_img = nib.load(flash_mask.path)
    topup_fmap_img = nib.load(topup_fmap.path)
    medic_fmap_img = nib.load(medic_fmap.path)
    func_mask_img = nib.load(func_mask.path)

    # now mask the data
    flash_fmap_data = flash_fmap_img.get_fdata() * flash_mask_img.get_fdata()
    topup_fmap_data = topup_fmap_img.get_fdata() * func_mask_img.get_fdata()
    medic_fmap_data = medic_fmap_img.get_fdata() * func_mask_img.get_fdata()

    # 
