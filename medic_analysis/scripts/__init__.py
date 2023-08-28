import argparse
import warnings
from pathlib import Path

# ignore future warnings
warnings.simplefilter("ignore", FutureWarning)

# create a default parser
parser = argparse.ArgumentParser(description="Andrew Van, vanandrew@wustl.edu, 2023")
parser.add_argument("--bids_dir", help="Path to the BIDS dataset. Uses default paths if None specified.")
parser.add_argument(
    "--output_dir",
    help="Path to the output directory. Dumps outputs to derivatives folder of bids path if none specified.",
)


# phase encoding direction string
PED_TABLE = {
    "i": "1 0 0",
    "i-": "-1 0 0",
    "j": "0 1 0",
    "j-": "0 -1 0",
    "k": "0 0 1",
    "k-": "0 0 -1",
}

# Set polarity index
POLARITY_IDX = {"PA": 0, "AP": 1}

# output directories
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR = Path(__file__).resolve().parent.parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Inch to mm conversion
INCHES_TO_MM = 25.4
MM_TO_INCHES = 1 / INCHES_TO_MM
