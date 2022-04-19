# %%
from analyze_mitgcm import plotMIT, toolsMIT
import numpy as np
import pickle
import importlib
import warnings

warnings.filterwarnings("ignore")


path = [
    "2lay_1SLOPE",
    "2lay_1SLOPE_NUe-4",
    "2lay_1SLOPE_NUe-5",
    # "2lay_1SLOPE_nAW25",
    "2lay_1SLOPE_nAW20",
    # "2lay_1SLOPE_nAW15",
    "2lay_1SLOPE_nAW10",
    # "2lay_1SLOPE_nAW05",
    "2lay_1SLOPE_AW00",
    "2lay_1SLOPE_AW10",
    # "2lay_1SLOPE_AW15",
    "2lay_1SLOPE_AW20",
    # "2lay_1SLOPE_AW25",
    "2lay_1SLOPE_AW30",
    # "2lay_1SLOPE_AW35",
    "2lay_1SLOPE_AW40",
    # "2lay_1SLOPE_AW45",
    "2lay_1SLOPE_AW50",
    # "2lay_1SLOPE_AW55",
    "2lay_1SLOPE_AW60",
    "100km_2lay_1SLOPE_AW60",
    "sgd_2lay_1SLOPE_nAW20",
    "sgd_2lay_1SLOPE_nAW10",
    "sgd_2lay_1SLOPE_AW00",
    "sgd_2lay_1SLOPE",
    "sgd_2lay_1SLOPE_AW10",
    "sgd_2lay_1SLOPE_AW20",
    "sgd_2lay_1SLOPE_AW30",
    "sgd_2lay_1SLOPE_AW40",
    "sgd_2lay_1SLOPE_AW50",
    "sgd_2lay_1SLOPE_AW60",
    "100km_2lay_1SLOPE_AW60",
]
# path = np.flip(path)

prx = [25e3]

# %%
# load

importlib.reload(plotMIT)
toolsMIT.load_all(path, "final")


# %%
# Plume

importlib.reload(plotMIT)
plotMIT.plot_plume("_plume_all.png", path, sec=False, which=["sum"])

# %%
