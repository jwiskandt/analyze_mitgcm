from telnetlib import SB
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.ndimage.filters import uniform_filter1d
from . import loadMIT
from . import toolsMIT as TM
import pandas as pd


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def ot_time(upr, Lx, dz, dy):
    if np.nansum(upr) == 0:
        mot = otp = otn = 0
        return mot, otp, otn

    negi = np.nonzero(upr < 0)[0]
    posi = np.nonzero(upr > 0)[0]
    maxi = np.nonzero(upr == np.nanmax(upr))[0]
    mini = np.nonzero(upr == np.nanmin(upr))[0]

    if maxi > np.max(negi):
        maxido = -1
    else:
        maxido = negi[np.min(np.nonzero(negi > maxi[0])[0])]

    if maxi < np.min(negi):
        maxiup = 0
    else:
        maxiup = negi[np.max(np.nonzero(negi < maxi[0])[0])] + 1

    if np.max(mini) > np.max(posi):
        minido = -1
    else:
        minido = posi[np.min(np.nonzero(posi > mini[0])[0])]

    if np.min(mini) < np.min(posi):
        miniup = 0
    else:
        miniup = posi[np.max(np.nonzero(posi < mini[0])[0])] + 1

    upos = np.nanmean(upr[maxiup:maxido])
    uneg = np.nanmean(upr[miniup:minido])

    otp = np.abs(Lx / np.abs(upos) / 86400)
    otn = np.abs(Lx / np.abs(uneg) / 86400)
    mot = otn + otp

    return mot, otn, otp


def ot_time_Q(coords, upr, Qout=False):
    dz = coords["dz"][0]
    dy = 10
    dx = coords["dx"][0]
    V = np.sum(coords["hfac"][:, 0:-200] * dx * dy * dz)
    Q = np.nancumsum(upr * dy * dz)
    mot = np.abs(V / np.max(np.abs(Q)) / 86400)
    if Qout:
        return mot, Q
    else:
        return mot


def gadeline(tice=-2, tam=0, sam=35, z=0):

    g = 9.81
    rhonil = 9.998e2
    a0 = -0.0575
    c0 = 0.0901
    b = -7.61e-8
    tfr1 = a0 * 0
    tfr2 = b * rhonil * g * z
    tfr = tfr1 + c0 + tfr2

    # %% Gade Line & T-Freeze
    L = 334e3
    cp = 3.994e3
    ci = 2e3
    gam = ci / cp

    T0 = tice  # ice
    T1 = tfr
    # Solve for T2
    T3 = tam

    TF = T3 - T1
    # S1 no exist
    S2 = sam - 2
    S3 = sam
    dTdS = S3 ** (-1) * (TF + L / cp + gam * (T1 - T0))

    T2 = T3 - dTdS * (S3 - S2)
    return [S2, S3], [T2, T3]


def calc_dist_along(a, b, c=None):
    if c is None:
        c = np.zeros(np.array(np.shape(a)))

    d = np.zeros(np.array(np.shape(a)))

    d[1:] = np.cumsum(np.sqrt((a[1:] - a[0:-1]) ** 2 + (b[1:] - b[0:-1]) ** 2))
    d[0] = 0

    return d


def nanave2d(a, w, axis=0):

    if axis == 0:
        niter = np.shape(a)[1]
        ave = np.zeros(niter)

        for i in np.arange(0, niter):
            if np.nansum(w[:, i]) == 0:
                ave[i] = np.nan
            else:
                ave[i] = np.nansum(a[:, i] * w[:, i]) / np.nansum(w[:, i])
    else:
        niter = np.shape(a)[0]
        ave = np.zeros(niter)

        for i in np.arange(0, niter):
            if np.nansum(w[i, :]) == 0:
                ave[i] = np.nan
            else:
                ave[i] = np.nansum(a[i, :] * w[i, :]) / np.nansum(w[i, :])

    return ave


def plume(coords, var, ret=[], mask="w"):
    """
    calculate Plume diagnostics
    """

    hfac = 1 * coords["hfac"]
    ice = coords["ice"]

    x = coords["x"]
    dx = coords["dx"]
    if len(ice) > len(x):
        ice = ice[: len(x)]

    d = calc_dist_along(x, ice) + x[0]

    z = coords["z"]
    dz = -coords["dz"][0]

    pthresh = 35 * dz
    z_plu = np.arange(0, -pthresh - dz, -dz)

    t = np.nanmean(var["t_all"], axis=0)
    taw = var["tref"][-1]
    tref = var["tref"]
    tamb = t[:, 2000]
    dt = np.zeros(np.shape(t))
    # dt = t - taw
    s = np.nanmean(var["s_all"], axis=0)
    saw = var["sref"][-1]
    sref = var["sref"]
    samb = s[:, 2000]
    ds = np.zeros(np.shape(s))
    for i in np.arange(0, len(x)):
        ds[:, i] = s[:, i] - sref
        dt[:, i] = t[:, i] - tref
    # ds = s - saw
    u = np.nanmean(var["u_all"], axis=0)
    u_all = var["u_all"]
    w_all = var["w_all"]
    w = np.nanmean(var["w_all"], axis=0)

    # fig, ax = plt.subplots(1, 1)
    # cf = ax.contourf(ds)
    # fig.colorbar(cf, ax=ax)
    # fig.show()

    pmask = np.ones(np.shape(t)) * hfac
    for i in np.arange(0, np.shape(t)[1]):
        if ice[i] < 0:
            indi = np.where(z < (ice[i] - pthresh))
            pmask[indi, i] = 0
        else:
            pmask[:, i] = 0

    u, w = loadMIT.uw_ontracer(u, w, coords)

    wthresh = 0.0000
    tthresh = -0.01
    sthresh = -0.01

    if mask == "w":
        pmask[w <= wthresh] = 0
    elif mask == "u":
        pmask[u <= wthresh] = 0
    elif mask == "t":
        pmask[dt >= tthresh] = 0
    elif mask == "s":
        pmask[ds >= sthresh] = 0

    t_plu = np.zeros([int(pthresh / dz) + 1, np.shape(d)[0]]) * np.nan
    dt_plu = np.zeros([int(pthresh / dz) + 1, np.shape(d)[0]]) * np.nan
    s_plu = np.zeros([int(pthresh / dz) + 1, np.shape(d)[0]]) * np.nan
    u_plu = np.zeros([int(pthresh / dz) + 1, np.shape(d)[0]]) * np.nan
    w_plu = np.zeros([int(pthresh / dz) + 1, np.shape(d)[0]]) * np.nan
    u_plu_all = (
        np.zeros([np.shape(u_all)[0], int(pthresh / dz) + 1, np.shape(d)[0]]) * np.nan
    )
    w_plu_all = (
        np.zeros([np.shape(w_all)[0], int(pthresh / dz) + 1, np.shape(d)[0]]) * np.nan
    )

    for i in np.arange(np.shape(d)[0]):
        plu = np.where(pmask[:, i] != 0)
        t_plu[: np.shape(plu)[1], i] = t[plu, i]
        dt_plu[: np.shape(plu)[1], i] = dt[plu, i]
        s_plu[: np.shape(plu)[1], i] = s[plu, i]
        u_plu[: np.shape(plu)[1], i] = u[plu, i]
        w_plu[: np.shape(plu)[1], i] = w[plu, i]
        for n in range(np.shape(u_all)[0]):
            u_plu_all[n, : np.shape(plu)[1], i] = u_all[n, plu, i]
            w_plu_all[n, : np.shape(plu)[1], i] = w_all[n, plu, i]

    thick = np.nansum(dz * pmask, axis=0)

    if "flx" in ret:
        uflx = u * hfac * dz
        wflx = w * dx

        flx = np.sqrt(uflx**2 + wflx**2)

        uflx = np.nansum(uflx, axis=0)
        wflx = np.nansum(wflx, axis=0)
        flx = np.nansum(flx, axis=0)

    ret_dic = {
        "d": d,
        "thick": thick,
        "t_plu": t_plu,
        "dt_plu": dt_plu,
        "s_plu": s_plu,
        "u_plu": u_plu,
        "w_plu": w_plu,
        "u_plu_all": u_plu_all,
        "w_plu_all": w_plu_all,
        "z_plu": z_plu,
        "dt": dt,
        "ds": ds,
    }
    if "flx" in ret:
        ret_dic.update(
            {
                "flx": flx,
                "uflx": uflx,
                "wflx": wflx,
            }
        )

    return ret_dic


def rho_cont(T, S):
    t = np.linspace(min(T), max(T), 99)
    s = np.linspace(min(S), max(S), 100)

    tm, sm = np.meshgrid(t, s)

    t0 = 1
    s0 = 35
    tAlpha = -0.4e-4
    sBeta = 8.0e-4
    rnil = 999.8

    rm = rnil * (1 + tAlpha * (tm - t0) + sBeta * (sm - s0))
    return tm, sm, rm


def extract_mdata():

    global mdata
    coords = TM.coords
    dx = coords[0]["dx"][-1]
    ice = coords[0]["ice"]
    data = TM.data
    SHIflx = TM.SHIflx

    lam1 = -5.75e-2
    lam2 = 9.01e-2
    lam3 = -7.61e-4

    mdata = {"T_AW": [], "MeltFlux": [], "PlumeFlux": [], "TF": [], "AveMelt": []}
    for vi in np.arange(0, np.shape(data)[0]):

        plumew = plume(coords[vi], data[vi], ["flx"], mask="w")
        flx = plumew["flx"]

        taw = data[vi]["tref"][-1]

        t_ice = plumew["t_plu"][0, :]
        s_ice = plumew["s_plu"][0, :]
        si_fil = uniform_filter1d(s_ice, size=20)

        t_f = lam1 * si_fil + lam2 + ice * lam3
        TF = t_ice - t_f

        melt = np.abs(SHIflx[vi]["fwfx"]) / 1000 * dx

        cmelt = np.nancumsum(melt)
        amelt = np.nanmean(melt)
        # safe data for fitting
        mdata["T_AW"].append(taw)
        mdata["MeltFlux"].append(cmelt[np.nanargmax(flx)])
        mdata["AveMelt"].append(amelt)
        mdata["PlumeFlux"].append(np.nanmax(flx))
        mdata["TF"].append(np.nanmean(TF[0 : np.nanargmax(flx)]))

    mdata = pd.DataFrame(mdata)


def extract_prdata(path, prx=21e3, to_file=False):

    global prdata
    coords = TM.coords
    z = coords[0]["z"]
    data = TM.data

    prdata = {"z": z}
    prdf = pd.DataFrame(prdata)
    print(prdf.head())
    for vi in range(np.shape(data)[0]):
        pri = np.min(np.where(coords[vi]["x"] >= prx)[0])
        tpr = TM.data[vi]["t_all"][vi, :, pri]
        spr = TM.data[vi]["s_all"][vi, :, pri]

        prdf["t_{}".format(path[vi])] = tpr
        prdf["s_{}".format(path[vi])] = spr

        print(prdf.head())

    if to_file:
        prdf.to_csv("profile_ts_{:.0f}km.csv".format(prx / 1e3), index=False)
