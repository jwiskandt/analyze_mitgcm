from . import loadMIT
import numpy as np
import pickle


def timeseries(path, start, stop, step, which=[], freq=360):

    for i in np.arange(0, np.shape(path)[0]):
        varfile = "mydata/" + path[i] + "/var_times.pkl"
        coordsfile = "mydata/" + path[i] + "/coords.pkl"
        SHIflxfile = "mydata/" + path[i] + "/SHIflx_times.pkl"
        gammafile = "mydata/" + path[i] + "/gamma_times.pkl"
        print("load " + path[i])

        coords = loadMIT.load_coord(path[i])
        with open(coordsfile, "wb") as a_file:
            pickle.dump(coords, a_file)
        if "tsu" in which or not which:
            var = loadMIT.load_tsu(coords, path[i], start, stop, step, freq=freq)
            with open(varfile, "wb") as a_file:
                pickle.dump(var, a_file)
        if "SHI" in which or not which:
            SHIflx = loadMIT.load_SHIflux(coords, path[i], start, stop, step, freq=freq)
            with open(SHIflxfile, "wb") as a_file:
                pickle.dump(SHIflx, a_file)
        if "gamma" in which or not which:
            gamma = loadMIT.load_gamma(path[i], start, stop, step, freq=freq)
            with open(gammafile, "wb") as a_file:
                pickle.dump(gamma, a_file)


def timeave(path, start, stop, which=[]):
    range = stop - start
    for i in np.arange(0, np.shape(path)[0]):
        varfile = "mydata/" + path[i] + "/var_final.pkl"
        coordsfile = "mydata/" + path[i] + "/coords.pkl"
        SHIflxfile = "mydata/" + path[i] + "/SHIflx_final.pkl"
        gammafile = "mydata/" + path[i] + "/gamma_final.pkl"
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


def create(path, start, stop, which=[]):
    range = stop - start
    for i in np.arange(0, np.shape(path)[0]):
        varfile = "mydata/" + path[i] + "/var_final.pkl"
        coordsfile = "mydata/" + path[i] + "/coords.pkl"
        SHIflxfile = "mydata/" + path[i] + "/SHIflx_final.pkl"
        gammafile = "mydata/" + path[i] + "/gamma_final.pkl"
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
