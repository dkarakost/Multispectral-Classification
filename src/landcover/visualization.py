import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# copy your color_map_entries dictionary here
COLOR_MAP = {
    0: ("#D20000", "Artificial surfaces"),
    1: ("#FDD327", "Cultivated areas"),
    2: ("#239800", "Broadleaf tree cover"),
    3: ("#085E00", "Coniferous tree cover"),
    4: ("#F99627", "Herbaceous vegetation"),
    5: ("#8D8B00", "Moors and Heathland"),
    6: ("#956DFF", "Marshes"),
    7: ("#956DC4", "Peatbogs"),
    8: ("#9A9A9A", "Natural material surfaces"),
    9: ("#1445F9", "Water bodies"),
}
CMAP = ListedColormap([c for c, _ in COLOR_MAP.values()])


def plot_map(class_map, title, ax=None):
    """class_map is (height, width) with values 0-9."""
    if ax is None:
        ax = plt.gca()
    ax.imshow(class_map, cmap=CMAP, vmin=0, vmax=len(COLOR_MAP) - 1, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")


def plot_legend():
    handles = [Patch(color=c, label=n) for c, n in COLOR_MAP.values()]
    plt.figure(figsize=(0.1, 0.1))
    plt.legend(handles=handles, ncols=2, loc="upper right")
    plt.axis("off")


def plot_confusion(cm, class_ids, title="Confusion matrix"):
    """cm should be the normalised matrix from evaluate.normalized_confusion."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_ids, yticklabels=class_ids)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()