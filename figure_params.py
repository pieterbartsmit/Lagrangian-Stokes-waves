"""
Some simple default settings for figure generation
"""

import matplotlib.pyplot as plt
params = {
    "axes.labelsize": 12,
    "axes.labelcolor": "grey",
    "font.size": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 10,
    "xtick.color": "grey",
    "ytick.color": "grey",
    "ytick.labelsize": 10,
    "text.usetex": False,
    "font.family": "sans-serif",
    "axes.grid": False,
}

plt.rcParams.update(params)