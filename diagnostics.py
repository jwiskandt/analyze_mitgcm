from telnetlib import SB
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from . import loadMIT


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


def plume(coords, var, ret):
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

    pthresh = 40 * dz
    z_plu = np.arange(0, -dz * 41, -dz)

    t = np.nanmean(var["t_all"], axis=0)
    s = np.nanmean(var["s_all"], axis=0)
    u = np.nanmean(var["u_all"], axis=0)
    w = np.nanmean(var["w_all"], axis=0)

    pmask = np.ones(np.shape(t)) * hfac
    for i in np.arange(0, np.shape(t)[1]):
        if ice[i] < 0:
            indi = np.where(z < (ice[i] - pthresh))
            pmask[indi, i] = 0
        else:
            pmask[:, i] = 0

    u, w = loadMIT.uw_ontracer(u, w, coords)

    wthresh = 0
    pmask[u < wthresh] = 0
    t_plu = np.zeros([int(pthresh / dz) + 1, np.shape(d)[0]]) * np.nan
    s_plu = np.zeros([int(pthresh / dz) + 1, np.shape(d)[0]]) * np.nan
    u_plu = np.zeros([int(pthresh / dz) + 1, np.shape(d)[0]]) * np.nan
    w_plu = np.zeros([int(pthresh / dz) + 1, np.shape(d)[0]]) * np.nan

    for i in np.arange(np.shape(d)[0]):
        plu = np.where(pmask[:, i] != 0)
        t_plu[: np.shape(plu)[1], i] = t[plu, i]
        s_plu[: np.shape(plu)[1], i] = s[plu, i]
        u_plu[: np.shape(plu)[1], i] = u[plu, i]
        w_plu[: np.shape(plu)[1], i] = w[plu, i]

    t = t * pmask
    s = s * pmask
    u = u * pmask
    w = w * pmask
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
        "s_plu": s_plu,
        "u_plu": u_plu,
        "w_plu": w_plu,
        "z_plu": z_plu,
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
