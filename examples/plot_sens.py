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
    "2lay_1SLOPE_nAW20",
    "2lay_1SLOPE_AW60",
    "2lay_1SLOPE_AW20",
    "2lay_1SLOPE_AW40",
    "100km_2lay_1SLOPE_AW60",
    "sgd_2lay_1SLOPE_nAW20",
    "sgd_2lay_1SLOPE",
    "sgd_2lay_1SLOPE_AW20",
    "sgd_2lay_1SLOPE_AW40",
    "sgd_2lay_1SLOPE_AW60",
    "100km_2lay_1SLOPE_AW60",
]
# path = np.flip(path)
prx = [25e3]

# %%
# load

importlib.reload(plotMIT)
toolsMIT.load_all(path, "final")
path = toolsMIT.short_path(path)


# %%
# Plume

importlib.reload(plotMIT)
importlib.reload(toolsMIT)
plotMIT.plot_plume("_plume_sens.png", path, sec=False, which=["sum"])

# %%
