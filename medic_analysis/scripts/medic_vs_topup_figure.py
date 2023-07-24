from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from medic_analysis.common import FIGURE_OUT


sns.set_theme(style="white")

UPENN_DIR = Path("/home/usr/vana/GMT2/Andrew/UPenn")


def main():
    # load in images from UPenn directory
    medic_img = Image.open(UPENN_DIR / "medic.png")
    topup_img = Image.open(UPENN_DIR / "topup.png")

    # create a figure with two subplots (constrained layout)
    f = plt.figure(figsize=(14, 9), layout="constrained")
    ax1 = f.add_subplot(2, 1, 1)
    ax1.imshow(topup_img, cmap="gray")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.text(5, 5, "(A) TOPUP", color="white", fontsize=18, verticalalignment="top")
    ax2 = f.add_subplot(2, 1, 2)
    ax2.imshow(medic_img, cmap="gray")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.text(5, 5, "(B) MEDIC", color="white", fontsize=18, verticalalignment="top")
    f.savefig(FIGURE_OUT / "medic_vs_topup.png", dpi=300, bbox_inches="tight")
