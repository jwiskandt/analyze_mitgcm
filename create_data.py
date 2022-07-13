from . import loadMIT
import numpy as np
import pickle
import gc


def create(path, start, stop, which=[]):
    for i in np.arange(0, np.shape(path)[0]):
        varfile = "mydata/{path}/var_{start}_{stop}.pkl".format(
            path=path[i], start=start, stop=stop
        )
        SHIflxfile = "mydata/{path}/SHIflx_{start}_{stop}.pkl".format(
            path=path[i], start=start, stop=stop
        )
        gammafile = "mydata/{path}/gamma_{start}_{stop}.pkl".format(
            path=path[i], start=start, stop=stop
        )
        coordsfile = "mydata/" + path[i] + "/coords.pkl"
        coords = loadMIT.load_coord(path[i])
        with open(coordsfile, "wb") as a_file:
            pickle.dump(coords, a_file)
        if "tsu" in which or not which:
            var = loadMIT.load_tsu(path[i], start, stop)
            with open(varfile, "wb") as a_file:
                pickle.dump(var, a_file)
        if "SHI" in which or not which:
            SHIflx = loadMIT.load_SHIflux(coords, path[i], start, stop)
            with open(SHIflxfile, "wb") as a_file:
                pickle.dump(SHIflx, a_file)
        if "gamma" in which or not which:
            gamma = loadMIT.load_gamma(path[i], start, stop)
            with open(gammafile, "wb") as a_file:
                pickle.dump(gamma, a_file)

    gc.collect()
