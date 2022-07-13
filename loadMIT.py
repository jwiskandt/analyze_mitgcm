from MITgcmutils import mds
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
import os


def load_coord(path):
    x = mds.rdmds(path + "/XC")
    x = x[0, :]
    x = np.array(x)
    xg = mds.rdmds(path + "/XG")
    xg = xg[0, :]
    dx = np.gradient(x)
    nx = np.shape(x)[0]

    z = mds.rdmds(path + "/RC")
    z = z[:, 0, 0]
    z = np.array(z)
    zg = mds.rdmds(path + "/RF")
    zg = zg[0:-1, 0, 0]
    dz = np.gradient(z)
    nz = np.shape(z)[0]

    hfac = np.squeeze(np.array(mds.rdmds(path + "/hFacC")))

    topo = np.fromfile(path + "/topog.slope", dtype=">f8")
    ice = np.fromfile(path + "/icetopo.exp1", dtype=">f8")

    return {
        "x": x,
        "xg": xg,
        "dx": dx,
        "nx": nx,
        "z": z,
        "zg": zg,
        "dz": dz,
        "nz": nz,
        "ice": ice,
        "topo": topo,
        "hfac": hfac,
    }


def load_tsu(coords, path, start, stop, step, freq=[]):
    if freq == []:
        freq = 8640

    files = [f for f in os.listdir(path) if ("dynDiag" in f) and ("data" in f)]
    files = [s.split(".")[1] for s in files]
    files.sort(key=int)

    # steps = np.arange(freq * start, freq * stop + 1, step * 1)
    files = [int(f) for f in files]
    step = files[1] - files[0]
    dt = 86400 / step
    steps = [s for s in files if (s >= start * step) and (s < stop * step)]
    print(steps)
    print(
        " ** load T, S, U from {} ** {}:{}:{}".format(
            path, start * freq, step, stop * freq
        )
    )
    time = np.zeros(np.shape(steps)[0])
    nt = np.shape(steps)[0]

    tref = np.fromfile(path + "/T.bound", dtype=">f8")
    print(tref[-1])
    tinit = np.fromfile(path + "/T.init", dtype=">f8")
    sref = np.fromfile(path + "/S.bound", dtype=">f8")
    sinit = np.fromfile(path + "/S.init", dtype=">f8")

    t_all = []
    s_all = []
    u_all = []
    w_all = []

    # loop through data/time steps
    for n in np.arange(0, nt):
        time[n] = steps[n] * dt
        print(" -- {} / {}".format(n + 1, nt))
        print(" -- Model time step: {} of {}".format(steps[n], np.max(steps)))
        print(
            " -- Model time:      {} hrs of {}".format(
                time[n] / 3600, np.max(steps) * dt / 3600
            )
        )
        n = n.astype("int")
        # %% load data
        data = mds.rdmds(path + "/dynDiag", steps[n])
        t_all.append(data[0, :, 0, :])
        s_all.append(data[1, :, 0, :])

        u_all.append(data[2, :, 0, :])
        w_all.append(data[3, :, 0, :])

    t_all = np.array(t_all)
    s_all = np.array(s_all)
    u_all = np.array(u_all)
    w_all = np.array(w_all)

    t_all[t_all == 0] = float("nan")
    s_all[s_all == 0] = float("nan")
    u_all[u_all == 0] = float("nan")
    w_all[w_all == 0] = float("nan")

    return {
        "t_all": t_all,
        "s_all": s_all,
        "u_all": u_all,
        "w_all": w_all,
        "tref": tref,
        "sref": sref,
        "tinit": tinit,
        "sinit": sinit,
        "time": time,
    }


def ave_tsu(t_all, s_all, u_all, w_all):
    aven = np.arange(0, np.shape(t_all)[0])
    t = np.nanmean(t_all[aven, :, :], axis=0)
    s = np.nanmean(s_all[aven, :, :], axis=0)
    u = np.nanmean(u_all[aven, :, :], axis=0)
    w = np.nanmean(w_all[aven, :, :], axis=0)

    return t, s, u, w


def uw_ontracer(u, w, coords):
    # interpolate u,w, on tracer points
    for i in np.arange(coords["nx"]):
        f = interpolate.interp1d(
            coords["zg"], w[:, i], fill_value="extrapolate"
        )
        w[:, i] = f(coords["z"])

    for j in np.arange(coords["nz"]):
        f = interpolate.interp1d(
            coords["xg"], u[j, :], fill_value="extrapolate"
        )
        u[j, :] = f(coords["x"])
    return u, w


def load_SHIflux(coords, path, start, stop, step, freq=[]):
    if freq == []:
        freq = 8640
    print(
        " ** load Shelf-ice Fluxes {} ** {}:{}:{}".format(
            path, start * freq, step, stop * freq
        )
    )
    step = step * freq
    steps = np.arange(freq * start, freq * stop + 1, step * 1)
    time = np.zeros(np.shape(steps)[0])
    dt = 10
    nt = np.shape(steps)[0]

    hef_all = []
    fwf_all = []
    Fh_all = []
    Fs_all = []
    fwf_mean = []

    for n in np.arange(0, nt):
        time[n] = steps[n] * dt
        print(" -- {} / {}".format(n + 1, nt))
        print(" -- Model time step: {} of {}".format(steps[n], np.max(steps)))
        print(
            " -- Model time:      {} hrs of {}".format(
                time[n] / 3600, np.max(steps) * dt / 3600
            )
        )
        step = steps[n]
        n = n.astype("int")

        SHIflux = mds.rdmds(path + "/SHIfluxDiag", step)

        fwf_all.append(SHIflux[0, :])
        hef_all.append(SHIflux[1, :])
        Fh_all.append(SHIflux[2, :])
        Fs_all.append(SHIflux[3, :])

    aven = np.arange(0, np.shape(hef_all)[0])
    fwf_all = np.array(fwf_all)
    hef_all = np.array(hef_all)
    Fs_all = np.array(Fs_all)
    Fh_all = np.array(Fh_all)

    fwf = np.nanmean(fwf_all[aven, 0, :], axis=0)
    fwf[fwf == -np.inf] = 0
    hef = np.nanmean(hef_all[aven, 0, :], axis=0)
    Fs = np.nanmean(Fs_all[aven, 0, :], axis=0)
    Fh = np.nanmean(Fh_all[aven, 0, :], axis=0)
    fwf = np.array(fwf)
    hef = np.array(hef)
    Fs = np.array(Fs)
    Fh = np.array(Fh)

    zsl = np.linspace(np.min(coords["ice"]), np.max(coords["ice"]), 85)
    zgrid = coords["ice"][0 : np.argmax(coords["ice"]) + 1]  # noqa: E203
    fwfgrid = fwf[0 : np.argmax(coords["ice"]) + 1]  # noqa: E203
    fwfgrid[np.isnan(fwfgrid)] = 0
    hefgrid = hef[0 : np.argmax(coords["ice"]) + 1]  # noqa: E203
    ffwf = interpolate.interp1d(zgrid, fwfgrid, kind="cubic")
    fhef = interpolate.interp1d(zgrid, hefgrid, kind="cubic")

    fwfsl = ffwf(zsl)
    hefsl = fhef(zsl)
    fwfsl = np.array(fwfsl)
    hefsl = np.array(hefsl)

    return {
        "fwfx": fwf,
        "fwf_all": fwf_all,
        "hefx": hef,
        "fwfz": fwfsl,
        "hefz": hefsl,
        "Fh": Fh,
        "Fs": Fs,
        "z": zsl,
    }


def load_gamma(path, start, stop, step, freq=[]):
    if freq == []:
        freq = 8640
    print(
        " ** load Shelf-ice Gammas {} ** {}:{}:{}".format(
            path, start * freq, step, stop * freq
        )
    )
    step = step * freq
    steps = np.arange(freq * start, freq * stop + 1, step * 1)
    steps = steps
    time = np.zeros(np.shape(steps)[0])
    dt = 10
    nt = np.shape(steps)[0]

    gammaS = []
    gammaT = []
    Ustar = []

    for n in np.arange(0, nt):
        time[n] = steps[n] * dt
        print(" -- {} / {}".format(n + 1, nt))
        print(" -- Model time step: {} of {}".format(steps[n], np.max(steps)))
        print(
            " -- Model time:      {} hrs of {}".format(
                time[n] / 3600, np.max(steps) * dt / 3600
            )
        )
        step = steps[n]
        n = n.astype("int")

        gamma = mds.rdmds(path + "/SHIgamma", step)

        gammaT.append(gamma[0, 0, :])
        gammaS.append(gamma[1, 0, :])
        Ustar.append(gamma[2, 0, :])

    return {
        "time": time,
        "Ustar": Ustar,
        "gammaT": gammaT,
        "gammaS": gammaS,
    }
