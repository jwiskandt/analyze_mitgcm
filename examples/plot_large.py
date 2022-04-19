# %%
from analyze_mitgcm import plotMIT, toolsMIT
import numpy as np
import pickle
import importlib
import warnings

warnings.filterwarnings("ignore")


path = [
    "100km_2lay_1SLOPE_AW60",
]

prx = [25e3]

importlib.reload(plotMIT)
toolsMIT.load_all(path, "final")

# %%
# plot sections

importlib.reload(plotMIT)

for i in np.arange(0, 1):
    plotMIT.plot_sec("_sec.png", path, i, prx)

# %%
# Profiles

importlib.reload(plotMIT)
plotMIT.plot_prof(
    "large_profiles.png",
    path,
    prx,
    Gade=False,
)


# %%
# Plume

importlib.reload(plotMIT)
plotMIT.plot_plume("_plume_all.png", path, sec=False, which=["sum"])

# %%
