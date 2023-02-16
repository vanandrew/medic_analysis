import argparse

# create a default parser
parser = argparse.ArgumentParser(description="Andrew Van, vanandrew@wustl.edu, 2023")
parser.add_argument("--bids_dir", help="Path to the BIDS dataset. Uses default paths if None specified.")
parser.add_argument(
    "--output_dir",
    help="Path to the output directory. Dumps outputs to derivatives folder of bids path if none specified.",
)
