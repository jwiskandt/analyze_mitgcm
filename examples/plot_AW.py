# %%
from analyze_mitgcm import plotMIT, toolsMIT
import numpy as np
import pickle
import importlib
import warnings

warnings.filterwarnings("ignore")


path = [
    "2lay_1SLOPE",
    "2lay_1SLOPE_nAW25",
    "2lay_1SLOPE_nAW20",
    "2lay_1SLOPE_nAW15",
    "2lay_1SLOPE_nAW10",
    "2lay_1SLOPE_nAW05",
    "2lay_1SLOPE_AW00",
    "2lay_1SLOPE_AW10",
    "2lay_1SLOPE_AW15",
    "2lay_1SLOPE_AW20",
    "2lay_1SLOPE_AW25",
    "2lay_1SLOPE_AW30",
    "2lay_1SLOPE_AW35",
    "2lay_1SLOPE_AW40",
    "2lay_1SLOPE_AW45",
    "2lay_1SLOPE_AW50",
    # "2lay_1SLOPE_AW50_JMD95",
    # "2lay_1SLOPE_AW50_TEOS10",
    "2lay_1SLOPE_AW55",
    "2lay_1SLOPE_AW60",
]
# path = np.flip(path)

prx = [25e3]

# %%
# load

importlib.reload(plotMIT)
toolsMIT.load_all(path, "final")
path = toolsMIT.short_path(path)


# %%
# plot sections

importlib.reload(plotMIT)

for i in np.arange(0, 1):
    plotMIT.plot_sec("_sec.png", path, i, prx)


# %%
# Profiles

importlib.reload(plotMIT)
plotMIT.plot_prof(
    "2lay_profiles.png",
    path,
    prx,
    Gade=False,
)

# %%
# Plume

importlib.reload(plotMIT)
plotMIT.plot_plume("_plume_AW.png", path, sec=False, which=["sum"])


# %% Gammas

# importlib.reload(plotMIT)
# plotMIT.plot_gamma("_gammas.png", path, coords, data, gammas)

# %% Forcings

# importlib.reload(plotMIT)
# plotMIT.plot_forcing("_forcings.png", path, coords, data, SHIflx)

# %% load and plot timeseries ##


from analyze_mitgcm import plotMIT
import numpy as np
import pickle
import importlib

# %%
path = [
    # "2lay_1SLOPE_nAW25",
    "2lay_1SLOPE_nAW20",
    # "2lay_1SLOPE_nAW15",
    "2lay_1SLOPE_nAW10",
    # "2lay_1SLOPE_nAW05",
    "2lay_1SLOPE_AW00",
    "2lay_1SLOPE",
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
]
# path = np.flip(path)

prx = [25e3]


# %%

importlib.reload(plotMIT)
plotMIT.plot_ekin(
    "_timeseries.png",
    path,
    prx=prx,
)
# %%
