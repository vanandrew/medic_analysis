from pathlib import Path
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from medic_analysis.common import FIGURE_OUT
import seaborn as sns

GROUP_TEMPLATE_ANALYSIS_PATH = Path("/data/egordon/data1/analysis/Evan/AA")
GROUP_TEMPLATE_ANALYSIS_OUTPUT = GROUP_TEMPLATE_ANALYSIS_PATH / "Medic_analysis.mat"

AA_DATA_DIR = Path("/data/Daenerys/ASD_ADHD/NP1173/derivatives/me_pipeline2")


def main():
    data = loadmat(str(GROUP_TEMPLATE_ANALYSIS_OUTPUT))

    # similarities
    similarities = data["similarities"].ravel()
    # get medic_topup labels
    medic_topup_labels = data["medic_topup"].ravel()
    medic_labels = medic_topup_labels == 1
    topup_labels = medic_topup_labels == 2
    # get medic and topup similarities
    medic_similarities = similarities[medic_labels]
    topup_similarities = similarities[topup_labels]

    # get where medic is better and topup is better
    medic_better = medic_similarities > topup_similarities
    topup_better = medic_similarities < topup_similarities

    # ensure size adds up
    assert medic_better.size + topup_better.size == len(similarities)

    # get similarities where medic is better and topup is better
    medic_better_similarities_medic = medic_similarities[medic_better]
    topup_better_similarities_medic = topup_similarities[medic_better]
    medic_better_similarities_topup = medic_similarities[topup_better]
    topup_better_similarities_topup = topup_similarities[topup_better]

    # plot group similarities
    f = plt.figure(figsize=(16, 9), layout="constrained")
    subfigs = f.subfigures(1, 2)
    subfigs[0].suptitle("(A) Similarity to ABCD Group Template")
    ax1 = subfigs[0].subplots(1, 1)
    sns.scatterplot(topup_better_similarities_topup, medic_better_similarities_topup, ax=ax1)
    sns.scatterplot(topup_better_similarities_medic, medic_better_similarities_medic, ax=ax1)
    ax1.axline((0, 0), slope=1, color="black", linestyle="--")
    ax1.set_xlabel("TOPUP Similarity (Correlation)")
    ax1.set_ylabel("MEDIC Similarity (Correlation)")
    ax1.text(0.5, 0.9, "MEDIC Better", transform=ax1.transAxes)
    ax1.text(0.5, 0.1, "TOPUP Better", transform=ax1.transAxes)
    x = np.array([-0.1, 0.7])
    y = x
    y2 = np.ones(x.shape) * 0.6
    y3 = np.zeros(x.shape)
    ax1.fill_between(x, y, y2, color="red", alpha=0.2)
    ax1.fill_between(x, y3, y, color="blue", alpha=0.2)
    ax1.set_xlim(0.0, 0.6)
    ax1.set_ylim(0.0, 0.6)

    # plot exemplar
    GROUP_TEMPLATE = AA_DATA_DIR / "ABCD_group_template_SCAN.png"
    MEDIC_SCAN = AA_DATA_DIR / "MEDIC_SCAN.png"
    TOPUP_SCAN = AA_DATA_DIR / "TOPUP_SCAN.png"
    group_img = Image.open(GROUP_TEMPLATE)
    medic_img = Image.open(MEDIC_SCAN)
    topup_img = Image.open(TOPUP_SCAN)
    subfigs[1].suptitle("(B) SCAN network similarity to ABCD Group Template")
    ax2 = subfigs[1].subplots(3, 1)
    ax2[0].imshow(group_img)
    ax2[0].set_xticks([])
    ax2[0].set_yticks([])
    ax2[0].set_title("ABCD Group Template (SCAN)", loc="center", y=-0.2)
    ax2[1].imshow(medic_img)
    ax2[1].set_xticks([])
    ax2[1].set_yticks([])
    ax2[1].set_title("MEDIC (SCAN)", loc="center", y=-0.2)
    ax2[2].imshow(topup_img)
    ax2[2].set_xticks([])
    ax2[2].set_yticks([])
    ax2[2].set_title("TOPUP (SCAN)", loc="center", y=-0.2)
    f.savefig(FIGURE_OUT / "group_template_compare.png", dpi=300, bbox_inches="tight")

    # show plot
    plt.show()
