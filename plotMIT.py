import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from scipy import interpolate
import pickle
from scipy.ndimage.filters import uniform_filter1d

# import gsw
import importlib
from ordered_set import OrderedSet

from . import loadMIT
from . import diagnostics

importlib.reload(diagnostics)


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


def get_expgeo(path):

    exp = np.zeros([np.shape(path)[0]], "int")
    geo = np.zeros([np.shape(path)[0]], "int")
    last = np.array([x[12:] for x in path])

    for vi in np.arange(0, np.shape(path)[0]):
        # How many layers

        if path[vi][0] == "1":
            geo[vi] = 0
        elif path[vi][0] == "2":
            geo[vi] = 2
        # how many slopes (if 2SLOPE add one to geo)
        if path[vi][5] == "2":
            geo[vi] += 1

        # variation from control
        if len(path[vi]) < 12:
            exp[vi] = 0
            last[vi] = "cont"

    lastset = list(OrderedSet(last))
    for n in lastset:
        indi = lastset.index(n)
        ns = [i for i in range(len(last)) if last[i] == n]
        exp[ns] = indi

    return exp, geo - 2


def load_colors():

    return cm.get_cmap("viridis_r")


def load_lines():

    lines = ["-", ":", "--", ":", "-."]
    markers = ["o", "x", "v", "^", "<", ">", "d"]
    return lines, markers


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


def plot_sec(figname, path, vi, prx):
    colors = load_colors()
    color = colors(0.9)
    tAlpha = (-0.4e-4,)
    sBeta = (8.0e-4,)
    rnil = 999.8

    (t, s, u, w) = loadMIT.ave_tsu(
        data[vi]["t_all"], data[vi]["s_all"], data[vi]["u_all"], data[vi]["w_all"]
    )
    x = coords[vi]["x"]
    z = coords[vi]["z"]
    ice = coords[vi]["ice"]
    dx = coords[vi]["dx"][0]
    dy = coords[vi]["dx"][0]
    dz = coords[vi]["dz"][0]
    time = data[vi]["time"] / 86400

    # nan above the ice
    for i in np.arange(1, coords[vi]["nx"]):
        t[z > ice[i], i] = float("nan")
        s[z > ice[i], i] = float("nan")
        u[z > ice[i], i] = float("nan")
        w[z > ice[i], i] = float("nan")

    psi = np.flipud(np.nancumsum(np.flip(u, axis=0) * dx, axis=0))
    print(np.shape(u))
    r = rnil * (1 + tAlpha * t + sBeta * s) - 1000
    np.shape(r)

    # tinit = np.squeeze(np.reshape(data[vi]["tinit"], np.shape(r)))
    # sinit = np.squeeze(np.reshape(data[vi]["sinit"], np.shape(r)))
    # rinit = rnil * (1 + tAlpha * tinit + sBeta * sinit) - 1000
    # levels = np.linspace(-1, 1, 41)

    dep = coords[vi]["topo"]
    icei = np.arange(0, np.shape(ice[ice < 0])[0] + 1)

    [fig, axs] = plt.subplots(1, 1, figsize=(12, 6))

    if prx is not None:
        for i in np.arange(0, np.shape(prx)[0]):
            axs.plot(np.ones(coords[vi]["nz"]) * prx[i] / 1000, z / 1000, "--k")

    cf0 = axs.contourf(
        x / 1000,
        z / 1000,
        r,
        32,  # np.linspace(-0.2, 0.2, 21),
        cmap="viridis_r",
        extend="both",
    )

    axs.contour(
        x / 1000,
        z / 1000,
        -psi,
        np.arange(2.5, 30, 2.5),
        colors="white",
        alpha=0.5,
    )
    axs.plot(x / 1000, dep / 1000, "k", linewidth=2)
    axs.plot(x[icei] / 1000, ice[icei] / 1000, "k", linewidth=2)
    axs.set_title("Exp: {} - Day {}-{}".format(path[vi], time[0], time[-1]))
    axs.set_xlabel("Distance from grounding line [km]")
    axs.set_ylabel("depth in [km]")

    if SHIflx[i] == 0:
        cbar = fig.colorbar(
            cf0, ax=axs, orientation="horizontal", fraction=0.1, anchor=(1.0, 0.1)
        )
    else:
        fwf = SHIflx[vi]["fwfx"]

        fwf[ice > 0] = np.nan
        melt = np.nansum(fwf) / 1000 * dz * dy
        print(melt)
        axs.text(
            1,
            -0.1,
            "Melt flux: \n{melt:7.3f} m^3/year".format(melt=melt),
            fontsize=14,
        )
        ax02 = axs.twinx()
        ax02.plot(x / 1000, fwf, color="green")
        ax02.set_ylabel("fresh water flux")
        ax02.tick_params(axis="y", colors="green")
        ax02.spines["right"].set_color("green")
        ax02.ticklabel_format(axis="y", style="sci", scilimits=[-1, 2])
        ax02.set_xlim(0, 30)
        ax02.set_ylim(2 * np.nanmin(fwf), 0)
        cbar = fig.colorbar(
            cf0, ax=ax02, orientation="horizontal", fraction=0.08, anchor=(1.0, 0.1)
        )
    cbar.set_label("Sigma")
    plt.tight_layout()
    fig.savefig("plots/" + path[vi] + figname, facecolor="white")
    plt.show()
    plt.close("all")


def plot_prof(fig_name, path, prx, Gade=False, SalT=True):
    colors = load_colors()
    lines, markers = load_lines()

    exp, geo = get_expgeo(path)

    t, s, u, w = init_uts(data)
    upr, spr, tpr = init_prof(coords[0], data, prx)
    dels = np.zeros(np.shape(upr))

    Q = np.zeros([np.shape(data)[0], np.shape(prx)[0], coords[0]["nz"]])
    mot = np.zeros([np.shape(data)[0], np.shape(prx)[0]])
    maxi = np.zeros([np.shape(data)[0], np.shape(prx)[0]])
    pri = np.zeros([np.shape(prx)[0]])
    sref = data[0]["sref"]
    dz = -coords[0]["dz"][0]

    for vi in np.arange(0, np.shape(data)[0]):
        t[vi, :, :], s[vi, :, :], u[vi, :, :], w[vi, :, :] = loadMIT.ave_tsu(
            data[vi]["t_all"], data[vi]["s_all"], data[vi]["u_all"], data[vi]["w_all"]
        )
        t = np.array(t)
        s = np.array(s)
        u = np.array(u)

        for i in np.arange(0, np.shape(prx)[0]):
            pri = np.min(np.where(coords[vi]["x"] >= prx[i])[0])
            upr[vi, i, :] = u[vi, :, pri]
            tpr[vi, i, :] = t[vi, :, pri]
            spr[vi, i, :] = s[vi, :, pri]
            dels[vi, i, :] = spr[vi, i, :] - sref
            cond = np.isnan(upr[vi, i, :]).all()
            maxi[vi, i] = np.nanargmax(upr[vi, i, :]) if not cond else np.nan
            mot[vi, i], Q[vi, i, :] = diagnostics.ot_time_Q(
                coords[vi], upr[vi, i], Qout=True
            )

    # sflx = (
    #     np.nansum(dels * upr, axis=2)
    #     / np.shape(upr)[2]
    #     / np.max(sref)
    #     * 3600
    #     * 24
    #     * 365
    # )
    sflx = np.nansum(dels * upr, axis=2) / np.max(sref) * dz
    tAlpha = (-0.4e-4,)
    sBeta = (8.0e-4,)
    mot = np.array(mot)
    maxi = np.array(maxi)
    Q = np.array(Q)
    upr = np.array(upr)
    tpr = np.array(tpr)
    spr = np.array(spr)

    fig = plt.figure(figsize=(12, 12))
    ax1 = plt.subplot(3, 2, 1)
    ax2 = plt.subplot(3, 2, 2)
    ax3 = plt.subplot(3, 2, 3)
    ax4 = plt.subplot(3, 2, 4)
    ax5 = plt.subplot(3, 2, 5)
    ax6 = plt.subplot(3, 2, 6)
    axs = [ax1, ax2, ax3, ax4, ax5, ax6]

    # axs[0].plot(var[0]["tref"], coords["z"] / 1000, "--k", label="tref")
    axs[0].grid("both")
    axs[0].set_ylim(-1.050, 0.050)
    axs[0].set_ylabel("depth [km]")
    axs[0].set_xlabel("Temperature Difference")

    # axs[1].plot(var[0]["sref"], coords["z"] / 1000, "--k", label="sref")
    axs[1].grid("both")
    axs[1].set_ylim(-1.050, 0.050)
    # axs[1].set_xlim(34.85, 35)
    axs[1].set_xlabel("Salinity Difference")

    axs[2].grid("both")
    axs[2].set_ylim(-1.050, 0.050)
    axs[2].set_xlim(27.490, 27.81)
    axs[2].set_xlabel("Sigma0")
    axs[2].set_ylabel("depth [km]")

    axs[3].grid("both")
    axs[3].set_ylim(-1.050, 0.050)
    axs[3].set_xlabel("hor. Velocity")

    axs[4].grid("both")
    # axs[4].set_ylim(-1.050, 0.050)
    axs[4].set_ylabel("integrated hor. fresh water transport")
    axs[4].set_xlabel("$T_{AW}$")

    axs[5].grid("both")
    axs[5].set_xlim(34.7, 35)
    axs[5].set_ylabel("Volume transport [$m^2/s$]")
    axs[5].set_xlabel("Salinity")

    # sam, ctm = np.meshgrid(
    #     np.arange(33.9, 35.1, 0.05),
    #     np.arange(-1.81, 5.5, 0.01),
    # )
    # dens = gsw.density.sigma0(sam, ctm)
    # densl = 999.8 * (1 + tAlpha * ctm + sBeta * sam) - 1000
    #
    # cf1 = axs[3].contour(
    #     sam,
    #     ctm,
    #     dens,
    #     (np.linspace(27, 28, 11)),
    #     colors="k",
    #     alpha=0.3,
    # )
    # cf2 = axs[3].contour(
    #     sam,
    #     ctm,
    #     densl,
    #     (np.linspace(27, 28, 11)),
    #     colors="k",
    #     alpha=0.6,
    #     linestyle="dashed",
    # )
    # axs[3].clabel(cf1)
    # axs[3].clabel(cf2)

    for vi in np.arange(0, np.shape(data)[0]):
        i = 0
        rpr = 999.8 * (1 + tAlpha * tpr[vi, i, :] + sBeta * spr[vi, i, :]) - 1000
        print(path[vi])
        rmax = rpr[np.nanargmax(upr[vi, i, :])]
        z = coords[vi]["z"]
        zmax = coords[vi]["z"][np.nanargmax(upr[vi, i, :])]
        color = colors(exp[vi] / np.max(exp))
        line = lines[geo[vi]]
        marker = markers[geo[vi]]

        tref = data[vi]["tref"]
        rpref = 999.8 * (1 + tAlpha * tref + sBeta * sref) - 1000

        if Gade:
            TG, SG = diagnostics.gadeline(tpr[vi, i, :], spr[vi, i, :], z)
            axs[3].plot(SG, TG, "--k", label="Gade ")

        for i in np.arange(0, np.shape(prx)[0]):
            # axs[0].plot(
            #     tref,
            #     coords["z"] / 1000,
            #     ":",
            #     color=color,
            # )
            axs[0].plot(
                tpr[vi, i, :] - tref,
                z / 1000,
                line,
                color=color,
            )
            axs[1].plot(
                spr[vi, i, :] - sref,
                z / 1000,
                line,
                color=color,
                label=path[vi],
            )
            axs[2].plot(
                rpr,
                z / 1000,
                line,
                color=color,
                label=path[vi],
            )
            axs[2].plot(
                rpref,
                z / 1000,
                "--",
                color=color,
            )
            axs[2].plot(
                rmax,
                zmax / 1000,
                marker,
                color=color,
            )
            # axs[3].plot(
            #     spr[vi, i, :],
            #     tpr[vi, i, :],
            #     marker,
            #     fillstyle="none",
            #     color=color,
            #     label=path[vi],
            # )
            axs[3].plot(
                upr[vi, i, :],
                z / 1000,
                line,
                color=color,
            )
            axs[4].plot(
                np.max(tref),
                -sflx[vi, i],
                marker,
                color=color,
            )
            axs[5].plot(
                spr[vi, i, :],
                upr[vi, i, :],
                line,
                color=color,
            )
            # axs[5].plot(
            #     sflx[vi, i, :],
            #     coords["z"] / 1000,
            #     line,
            #     color=color,
            # )

    plt.tight_layout()
    axs[1].legend(loc="lower center", bbox_to_anchor=(-0.1, 0), ncol=2)
    fig.savefig("plots/" + fig_name, facecolor="white")
    fig.show()


def plot_plume(figname, path, sec=False):
    importlib.reload(diagnostics)
    lam1 = -5.75e-2
    lam2 = 9.01e-2
    lam3 = -7.61e-4
    tAlpha = (-0.4e-4,)
    sBeta = (8.0e-4,)
    rnil = 999.8
    g = -9.81
    colors = load_colors()
    lines, markers = load_lines()
    z = coords[0]["z"]
    dx = coords[0]["dx"][0]

    exp, geo = get_expgeo(path)

    heights = [3, 3, 3, 3]
    fig1 = plt.figure(figsize=(8, 8))
    gs1 = GridSpec(4, 1, figure=fig1, height_ratios=heights)
    ax11 = fig1.add_subplot(gs1[0, :])
    ax12 = fig1.add_subplot(gs1[1, :])
    ax13 = fig1.add_subplot(gs1[2, :])
    ax14 = fig1.add_subplot(gs1[3, :])

    fig2 = plt.figure(figsize=(8, 6))
    gs2 = GridSpec(3, 3, figure=fig2)
    ax21 = fig2.add_subplot(gs2[0, :])
    ax22 = fig2.add_subplot(gs2[1, :])
    # ax23 = fig2.add_subplot(gs2[2, :])
    ax24 = fig2.add_subplot(gs2[2, 0])
    ax25 = fig2.add_subplot(gs2[2, 1])
    ax26 = fig2.add_subplot(gs2[2, 2])

    fig3 = plt.figure(figsize=(8, 6))
    gs3 = GridSpec(2, 3, figure=fig3)
    ax31 = fig3.add_subplot(gs3[0, :])
    ax32 = fig3.add_subplot(gs3[1, 0])
    ax33 = fig3.add_subplot(gs3[1, 1])
    ax34 = fig3.add_subplot(gs3[1, 2])

    for vi in np.arange(0, np.shape(data)[0]):
        plume = diagnostics.plume(coords[vi], data[vi], ["flx"], sec=sec)
        x = np.squeeze(plume["d"] / 1000)
        d = plume["thick"]
        taw = np.nanmax(data[vi]["tref"])
        ice = coords[vi]["ice"]
        tref = data[vi]["tref"][-1]
        sref = data[vi]["sref"][-1]

        melt = np.abs(SHIflx[vi]["fwfx"]) / 1000 * dx
        cmelt = np.nancumsum(melt)

        color = colors(exp[vi] / np.max(exp))
        line = lines[geo[vi]]
        marker = markers[geo[vi]]

        u_ice = plume["u_plu"][0, :]
        w_ice = plume["u_plu"][0, :]
        vel_ice = np.sqrt(u_ice**2 + w_ice**2)

        t_ice = plume["t_plu"][0, :]
        t_ave = np.nanmean(plume["t_plu"], axis=0)
        s_ice = plume["s_plu"][0, :]
        s_ave = np.nanmean(plume["s_plu"], axis=0)

        t_f = lam1 * s_ice + lam2 + ice * lam3

        nans, i = diagnostics.nan_helper(vel_ice)
        vel_ice[nans] = np.interp(i(nans), i(~nans), vel_ice[~nans])
        nans, i = diagnostics.nan_helper(s_ice)
        s_ice[nans] = np.interp(i(nans), i(~nans), s_ice[~nans])
        nans, i = diagnostics.nan_helper(s_ave)
        s_ave[nans] = np.interp(i(nans), i(~nans), s_ave[~nans])
        nans, i = diagnostics.nan_helper(t_ice)
        t_ice[nans] = np.interp(i(nans), i(~nans), t_ice[~nans])
        nans, i = diagnostics.nan_helper(t_ave)
        t_ave[nans] = np.interp(i(nans), i(~nans), t_ave[~nans])
        nans, i = diagnostics.nan_helper(d)
        d[nans] = np.interp(i(nans), i(~nans), d[~nans])

        sa_fil = uniform_filter1d(s_ave, size=20)
        si_fil = uniform_filter1d(s_ice, size=20)
        vel_fil = uniform_filter1d(vel_ice, size=20)
        ti_fil = uniform_filter1d(t_ice, size=20)
        ta_fil = uniform_filter1d(t_ave, size=20)
        d_fil = uniform_filter1d(d, size=20)

        s_buo = (sa_fil - sref) * sBeta * rnil * g
        t_buo = (ta_fil - tref) * tAlpha * rnil * g
        if vi == 0:
            s_b_ref = (sa_fil - sref) * 8e-4 * g
            t_b_ref = (ta_fil - tref) * -0.4e-4 * g
            t_force = t_ice - t_f

        flx = plume["flx"]

        print(np.shape(x))
        print(np.shape(d_fil))
        ax11.plot(x, d_fil, color=color)
        # ax13.plot(
        #    d, s_ice - 35, color=color, linestyle=line, alpha=0.1, label="_nolegend_"
        # )
        ax12.plot(
            x,
            si_fil - sref,
            color=color,
            linestyle=line,
            alpha=0.7,
        )
        ax13.plot(
            x,
            ti_fil - tref,
            color=color,
            linestyle=line,
            alpha=0.7,
            label=path[vi],
        )
        ax14.plot(x, vel_fil, color=color, linestyle=line, alpha=0.7, label=path[vi])
        ax14.plot(
            x, vel_ice, color=color, linestyle=line, alpha=0.1, label="_nolegend_"
        )

        # ------------Figure 2-------------

        ax21.plot(x, flx, color=color, linestyle=line, alpha=0.7, label=path[vi])
        ax22.plot(x, np.nancumsum(melt), color=color, linestyle=line, alpha=0.7)
        # ax23.plot(
        #    d,
        #    np.cumsum(-SHIflx[vi]["fwfx"] / 1000) / flx,
        #    color=color,
        #    linestyle=line,
        #    alpha=0.7,
        # )

        ax24.plot(taw, np.nanmax(flx), marker, color=color)
        ax25.plot(taw, cmelt[np.nanargmax(flx)], marker, color=color, label=path[vi])
        ax26.plot(taw, cmelt[np.nanargmax(flx)] / np.nanmax(flx), marker, color=color)

        print(taw)
        print(np.nanmax(flx))
        print(cmelt[np.nanargmax(flx)])
        print(cmelt[np.nanargmax(flx)] / np.nanmax(flx))

        # ------------Figure 3-------------

        if vi == 0:
            tlabel = "Temp"
            slabel = "Sal"
            label = "Total"
            ax31.plot(x, s_buo + 10, color="k", linestyle="-", alpha=0.7, label=label)
            ax31.plot(x, s_buo + 10, color="k", linestyle="--", alpha=0.7, label=tlabel)
            ax31.plot(x, s_buo + 10, color="k", linestyle=":", alpha=0.7, label=slabel)

        else:
            tlabel = "_nolegend"
            slabel = "_nolegend"
            label = "_nolegend"

        ax31.plot(
            x, s_buo + t_buo, color=color, linestyle=line, alpha=0.7, label=path[vi]
        )
        ax31.plot(x, s_buo, color=color, linestyle=":", alpha=0.7)
        ax31.plot(x, t_buo, color=color, linestyle="--", alpha=0.7)
        ax32.plot(
            tref,
            np.nanmean(t_buo[0 : np.nanargmax(flx)] + s_buo[0 : np.nanargmax(flx)]),
            marker,
            color=color,
        )
        ax33.plot(tref, np.nanmean(s_buo[0 : np.nanargmax(flx)]), marker, color=color)
        ax34.plot(tref, np.nanmean(t_buo[0 : np.nanargmax(flx)]), marker, color=color)

    ax11.set_ylabel("Plume \nthickness")
    ax11.grid("both")
    ax11.set_xlim(0, 20.5)

    ax12.set_xticklabels([])
    ax12.set_ylabel("$S_{BL}-S_{AW}$")
    ax12.grid("both")
    ax12.set_xlim(0, 20.5)
    ax12.set_ylim(-0.21, 0.01)
    ax12.ticklabel_format(axis="y", style="sci", scilimits=[-2, 2])

    ax13.set_xticklabels([])
    ax13.set_ylabel("$T_{BL}-T_{AW}$")
    ax13.grid("both")
    ax13.set_ylim(-1.21, 0.01)
    ax13.set_xlim(0, 20.5)

    ax14.set_ylabel("BL veloctiy")
    ax14.set_xlabel("distance along Ice")
    ax14.grid("both")
    ax14.set_xlim(0, 20.5)

    # ------------Figure 2-------------
    ax21.set_xticklabels([])
    ax21.set_ylabel("Plume Flux [m^2/s]")
    ax21.grid("both")
    ax21.set_xlim(0, 20.5)

    ax22.set_ylabel("melt from ice")
    ax22.set_xlabel("dist along ice")
    ax22.set_xticklabels([])
    ax22.grid("both")
    ax22.set_xlim(0, 20.5)

    # ax23.set_xlabel("dist along ice")
    # ax23.set_ylabel("cum(melt)/flx")
    # ax23.grid("both")
    # ax23.set_xlim(0, 20.5)
    # ax23.set_ylim(0, 0.001)
    # ax23.ticklabel_format(axis="y", style="sci", scilimits=[-1, 2])

    ax24.set_ylabel("Plume Vol. Flux \n(QP) [m^2/s]")
    ax24.set_xlabel("AW Temperature")
    ax24.grid("both")

    ax25.set_ylabel("Cummulative Melt \n(M)[m^2/s]")
    ax25.set_xlabel("AW Temperature")
    ax25.grid("both")

    ax26.set_ylabel("sum(M)/max(QP)")
    ax26.set_xlabel("AW Temperature")
    ax26.grid("both")
    ax26.ticklabel_format(axis="y", style="sci", scilimits=[-1, 2])

    ax31.set_xticklabels([])
    ax31.set_ylabel("Plume Buoyancy\ntotal")
    ax31.grid("both")
    ax31.set_xlim(0, 20.5)
    ax31.set_ylim(-0.25, 0.55)
    ax31.ticklabel_format(axis="y", style="sci", scilimits=[-1, 2])
    ax31.legend()

    ax32.set_ylabel("Plume Buoyancy\ntotal")
    ax32.set_ylim(0.19, 0.51)
    ax32.grid("both")

    ax33.set_ylabel("Sal")
    ax33.set_ylim(0.19, 0.51)
    ax33.grid("both")

    ax34.set_ylabel("Temp")
    ax34.set_ylim(-0.32, 0)
    ax34.grid("both")

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()

    ax14.legend(loc="center right", ncol=1)  # , bbox_to_anchor=(-0.00, 0))
    ax21.legend(loc="center right", ncol=1)  # , bbox_to_anchor=(-0.00, 0))
    fig1.savefig("plots/tsu" + figname, facecolor="white")
    fig2.savefig("plots/flux" + figname, facecolor="white")
    fig3.savefig("plots/buoy" + figname, facecolor="white")


def plot_ekin(figname, path, flx=[], prx=[]):

    pri = np.zeros(np.shape(prx)[0]).astype("int")
    lines, markers = load_lines()
    colors = load_colors()
    lines = np.array(lines)

    exp, geo = get_expgeo(path)

    [fig, axs] = plt.subplots(3, 1, figsize=(8, 5))

    print("start plotting {} timeseries".format(np.shape(path)[0]))
    for vi in np.arange(0, np.shape(path)[0]):
        print("plot {} of {}".format(vi, np.shape(path)[0]))
        data, coords, SHIflx, gammas = load_single(path[vi], "times")

        time = data["time"] / 86400
        upr = np.zeros([np.shape(data["u_all"])[0], coords["nz"]])

        color = colors(exp[vi] / np.max(exp))
        line = lines[geo[vi]]
        u = data["u_all"]
        w = data["w_all"]
        ekin = np.nansum(u * u + w * w, axis=(1, 2))
        motQ = np.zeros(np.shape(time))

        if SHIflx is None:
            melt = np.zeros(np.shape(ekin))
        else:
            melt = -(SHIflx["fwf_mean"] / 1000 * 86400 * 365)

        pri = int(str(min(np.where(coords["x"] >= prx)[0])))
        for n in np.arange(0, np.shape(data["time"])[0]):
            upr = data["u_all"][n, :, pri]
            motQ[n] = diagnostics.ot_time_Q(coords, upr)

        motQ = np.array(motQ)

        axs[0].plot(
            time,
            ekin,
            line,
            color=color,
            label="{}".format(path[vi]),
        )

        axs[1].plot(
            time,
            melt,
            line,
            color=color,
        )

        axs[2].plot(
            time,
            motQ,
            line,
            color=color,
        )

    axs[0].set_xticklabels([])
    axs[0].set_ylabel("Kinetic Energy")
    axs[0].grid("both")
    axs[0].legend(ncol=4, loc=8, fontsize="small")

    axs[1].set_xticklabels([])
    axs[1].set_ylabel("melt rate [m/a]")
    axs[1].grid("both")

    axs[2].set_xlabel("Model Days")
    axs[2].set_ylabel("Fjord Overturngin\n timescale")
    axs[2].grid("both")
    axs[2].set_ylim([18, 32])

    plt.tight_layout()
    plt.show()
    fig.savefig("plots/all" + figname, facecolor="white")
    plt.close("all")


def plot_gamma(figname, path, coords, var, gammas):
    lam1 = -5.75e-2
    lam2 = 9.01e-2
    lam3 = -7.61e-4
    colors = load_colors()
    lines, markers = load_lines()
    exp, geo = get_expgeo(path)

    fig = plt.figure(figsize=(12, 10))
    ax1 = plt.subplot(2, 3, 1)
    ax2 = plt.subplot(2, 3, 2)
    ax3 = plt.subplot(2, 3, 3)
    ax4 = plt.subplot(2, 1, 2)
    axs = [ax1, ax2, ax3, ax4]
    ax0 = axs[0].twiny()
    ax1 = axs[1].twiny()
    for vi in np.arange(0, np.shape(var)[0]):

        color = colors(exp[vi] / np.max(exp))
        line = lines[geo[vi]]

        ice1 = coords[vi]["ice"]
        ice = np.delete(ice1, ice1 > 0)

        gammaS = np.nanmean(gammas[vi]["gammaS"], axis=0)
        gammaSstd = np.nanstd(gammas[vi]["gammaS"], axis=0)
        gammaT = np.nanmean(gammas[vi]["gammaT"], axis=0)
        gammaTstd = np.nanstd(gammas[vi]["gammaT"], axis=0)
        # Ustar = np.nanmean(gammas[vi]["Ustar"], axis=0)

        gamS = np.delete(gammaS, ice1 > 0)
        gamT = np.delete(gammaT, ice1 > 0)
        gamSstd = np.delete(gammaSstd, ice1 > 0)
        gamTstd = np.delete(gammaTstd, ice1 > 0)
        # ust = np.delete(Ustar, ice1 > 0)

        plume = diagnostics.plume(coords[vi], var[vi], ["plume", "flx"])
        s = plume["s"][0, :]
        # t = plume["t"][0, :]
        taw = np.nanmax(var[vi]["tref"])
        # saw = np.nanmax(var[vi]["sref"])
        z = -plume["z"][0, :]
        flx = plume["flx"]
        s[z > z[np.argmax(flx)]] = np.nan
        gamS[ice > z[np.argmax(flx)]] = np.nan
        gamT[ice > z[np.argmax(flx)]] = np.nan
        print(path[vi])
        print(np.nanmean(gamS), np.nanmean(gamT))

        tF = lam1 * s + lam2 + lam3 * z

        fgam = interpolate.interp1d(ice, gamS, fill_value="extrapolate")
        gamSp = fgam(z)

        axs[0].plot(gamS, -ice, line, color=color)
        ax0.plot(gamSstd / gamS * 100, -ice, line, color=color, alpha=0.5)
        axs[1].plot(gamT, -ice, line, color=color)
        ax1.plot(gamTstd / gamT * 100, -ice, line, color=color, alpha=0.5)
        axs[2].plot(taw - tF, -z, line, color=color)
        axs[3].plot(taw - tF, gamSp, line, color=color, label=path[vi])

    axs[0].set_title("Gamma S")
    axs[0].set_ylabel("depth")
    axs[0].set_xlabel("Mean")
    ax0.set_xlabel("Std/Mean [%]")
    axs[0].set_ylim(1000, 0)
    # axs[0].set_xlim(0, 2.5e-6)

    axs[1].set_title("Gamma T")
    axs[1].set_xlabel("Mean")
    ax1.set_xlabel("Std/Mean [%]")
    axs[1].set_yticklabels([])
    axs[1].set_ylim(1000, 0)
    # axs[1].set_xlim(0, 2.5e-6)

    axs[2].set_title("$T_{AW} - T_{freeze}$")
    axs[2].set_ylim(1000, 0)
    axs[2].set_yticklabels([])

    axs[3].set_xlabel("$T_{AW} - T_{freeze}$")
    axs[3].set_ylabel("Gamma S")
    axs[3].grid("both")

    fig.tight_layout()
    axs[3].legend(ncol=6, loc="lower center", bbox_to_anchor=(0.5, 2))
    fig.savefig("plots/all" + figname, facecolor="white")


def plot_forcing(figname, path, coords, var, SHIflx):
    colors = load_colors()
    lines, markers = load_lines()
    exp, geo = get_expgeo(path)

    fig = plt.figure(figsize=(12, 10))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    axs = [ax1, ax2]

    for vi in np.arange(0, np.shape(var)[0]):

        color = colors(exp[vi] / np.max(exp))
        line = lines[geo[vi]]

        ice1 = coords[vi]["ice"]
        ice = np.delete(ice1, ice1 > 0)

        Fs1 = SHIflx[vi]["Fs"]
        # Fh1 = SHIflx[vi]["Fh"]

        Fs = np.delete(Fs1, ice1 > 0)
        # Fh = np.delete(Fh1, ice1 > 0)

        # plume = diagnostics.plume(coords[vi], var[vi], ["flx"])
        # z = -plume["z"][0, :]
        # flx = plume["flx"]
        # Fs[ice > z[np.argmax(flx)]] = np.nan
        # Fh[ice > z[np.argmax(flx)]] = np.nan

        axs[0].plot(Fs, -ice, line, color=color)
        axs[1].plot(Fs, -ice, line, color=color, label=path[vi])

    axs[0].set_title("Heat Forcing")
    axs[0].set_ylabel("depth")
    axs[0].set_ylim(1000, 0)
    # axs[0].set_xlim(0, 2.5e-6)

    axs[1].set_title("Fresh Water Forcing")
    axs[1].set_ylim(1000, 0)
    axs[1].set_ylabel("depth")
    axs[1].set_xlabel("$m^2/s^2$")
    axs[1].legend(ncol=3, loc=0)
    # axs[1].set_xlim(0, 2.5e-6)

    fig.tight_layout()
    fig.savefig("plots/all" + figname, facecolor="white")
