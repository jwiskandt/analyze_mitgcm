# %%
from analyze_mitgcm import plotMIT
import numpy as np
import pickle
import importlib
import warnings

warnings.filterwarnings("ignore")


path = [
    "2lay_1SLOPE",
    "2lay_1SLOPE_NUe-6",
    "2lay_1SLOPE_NUe-7",
]
# path = np.flip(path)

prx = [25e3]

# %%
# load

importlib.reload(plotMIT)
plotMIT.load_all(path, "final")

for i in np.arange(0, len(path)):
    if len(path[i]) < 12:
        path[i] = "control"
    elif "sgd" in path[i]:
        path[i] = path[i][16:] + "_SGD"
    else:
        path[i] = path[i][12:]


print(path)

# %%
# plot sections

importlib.reload(plotMIT)

for i in np.arange(0, 3):
    plotMIT.plot_sec("_sec_NU.png", path, i, prx)


# %%
# Profiles

importlib.reload(plotMIT)
plotMIT.plot_prof(
    "2lay_profiles_NU.png",
    path,
    prx,
    Gade=False,
)


# %%
