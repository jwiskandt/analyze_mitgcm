import numpy as np
import pickle
from matplotlib import cm


def load_path():
    return [
        "2lay_1SLOPE",
        "2lay_1SLOPE_NUe-4",
        "2lay_1SLOPE_NUe-5",
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
        "2lay_1SLOPE_AW55",
        "2lay_1SLOPE_AW60",
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
    ]


def short_path(path):
    for i in np.arange(0, len(path)):
        if len(path[i]) < 12:
            path[i] = "control"
        elif "sgd" in path[i]:
            path[i] = path[i][16:] + "_SGD"
        elif "100km" in path[i]:
            path[i] = path[i][18:] + "_100km"
        else:
            path[i] = path[i][12:]

    return path


def load_all(path, which):

    global data, coords, SHIflx, gammas

    print(path)
    path = list(path)
    data = np.zeros(np.shape(path))
    coords = np.zeros(np.shape(path))
    SHIflx = np.zeros(np.shape(path))
    gammas = np.zeros(np.shape(path))
    data = list(data)
    coords = list(coords)
    SHIflx = list(SHIflx)
    gammas = list(gammas)
    for i in np.arange(0, len(path)):
        print("loading {}".format(path[i]))
        varfile = "mydata/" + path[i] + "/var_{}.pkl".format(which)
        coordsfile = "mydata/" + path[i] + "/coords.pkl"
        SHIflxfile = "mydata/" + path[i] + "/SHIflx_{}.pkl".format(which)
        gammafile = "mydata/" + path[i] + "/gamma_{}.pkl".format(which)
        with open(varfile, "rb") as a_file:
            data[i] = pickle.load(a_file)
        with open(coordsfile, "rb") as a_file:
            coords[i] = pickle.load(a_file)
        with open(SHIflxfile, "rb") as a_file:
            SHIflx[i] = pickle.load(a_file)
        with open(gammafile, "rb") as a_file:
            gammas[i] = pickle.load(a_file)


def load_single(path, which):

    print("loading {}".format(path))
    varfile = "mydata/" + path + "/var_{}.pkl".format(which)
    coordsfile = "mydata/" + path + "/coords.pkl"
    SHIflxfile = "mydata/" + path + "/SHIflx_{}.pkl".format(which)
    gammafile = "mydata/" + path + "/gamma_{}.pkl".format(which)
    with open(varfile, "rb") as a_file:
        data = pickle.load(a_file)
    with open(coordsfile, "rb") as a_file:
        coords = pickle.load(a_file)
    with open(SHIflxfile, "rb") as a_file:
        SHIflx = pickle.load(a_file)
    with open(gammafile, "rb") as a_file:
        gammas = pickle.load(a_file)

    return data, coords, SHIflx, gammas


def identify(path, tref):

    lines = ["-", ":", "--", ":", "-."]
    markers = ["o", "x", "v", "^", "<", ">", "d"]
    colors = cm.get_cmap("cividis")
    colors2 = cm.get_cmap("copper")
    print(path)

    if "control" in path:
        marker = markers[-1]
        line = lines[-1]
        color = "k"
    elif "_SGD" in path:
        marker = markers[1]
        color = colors((tref + 2.5) / 10)
        line = lines[1]
        print("hier ist SGD")
    elif "NU" in path:
        marker = markers[2]
        color = colors2((int(path[-1]) - 4) / 2)
        line = lines[2]
    elif "100km" in path:
        marker = markers[3]
        color = colors((tref + 2.5) / 10)
        line = lines[3]
    else:
        marker = markers[0]
        color = colors((tref + 2.5) / 10)
        line = lines[0]

    return color, line, marker


def init_uts(var):

    u = np.zeros(
        [
            np.shape(var)[0],
            np.shape(var[0]["t_all"])[1],
            np.shape(var[0]["t_all"])[2],
        ]
    )
    w = np.zeros(
        [
            np.shape(var)[0],
            np.shape(var[0]["t_all"])[1],
            np.shape(var[0]["t_all"])[2],
        ]
    )
    t = np.zeros(
        [
            np.shape(var)[0],
            np.shape(var[0]["t_all"])[1],
            np.shape(var[0]["t_all"])[2],
        ]
    )
    s = np.zeros(
        [
            np.shape(var)[0],
            np.shape(var[0]["t_all"])[1],
            np.shape(var[0]["t_all"])[2],
        ]
    )

    return t, s, u, w


def init_prof(coords, var, prx):

    upr = np.zeros([np.shape(var)[0], np.shape(prx)[0], coords["nz"]])
    spr = np.zeros([np.shape(var)[0], np.shape(prx)[0], coords["nz"]])
    tpr = np.zeros([np.shape(var)[0], np.shape(prx)[0], coords["nz"]])

    return upr, spr, tpr
