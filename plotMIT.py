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
    path
    data = np.zeros(np.shape(path))
    coords = np.zeros(np.shape(path))
    SHIflx = np.zeros(np.shape(path))
    gammas = np.zeros(np.shape(path))
    data = list(data)
    coords = list(coords)
    SHIflx = list(SHIflx)
    gammas = list(gammas)
    for i in np.arange(0, np.shape(path)[0]):
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
    time = data[vi]["time"] / 86400

    # nan above the ice
    for i in np.arange(1, coords[vi]["nx"]):
        t[z > ice[i], i] = float("nan")
        s[z > ice[i], i] = float("nan")
        u[z > ice[i], i] = float("nan")
        w[z > ice[i], i] = float("nan")

    psi = np.nancumsum(u * dx, axis=0)
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
        psi,
        np.arange(2, 30, 2),
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
        melt = np.nanmean(fwf) / 1000 * 3600 * 24 * 365
        axs.text(
            1,
            -0.1,
            "Ave Meltrate: \n{melt:6.1f} m/year".format(melt=-melt),
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
    axs[3].set_xlabel("hor. Velocity streamfunction")

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


def plot_plume(figname, path):
    importlib.reload(diagnostics)
    colors = load_colors()
    lines, markers = load_lines()
    z = coords[0]["z"]
    dx = coords[0]["dx"][0]

    exp, geo = get_expgeo(path)

    heights = [1, 1, 3, 3, 3]
    fig1 = plt.figure(figsize=(8, 8))
    gs1 = GridSpec(5, 1, figure=fig1, height_ratios=heights)
    ax11 = fig1.add_subplot(gs1[0, :])
    ax12 = fig1.add_subplot(gs1[1, :])
    ax13 = fig1.add_subplot(gs1[2, :])
    ax14 = fig1.add_subplot(gs1[3, :])
    ax15 = fig1.add_subplot(gs1[4, :])

    ax11.plot(coords[0]["x"] / 1000, coords[0]["ice"], color="black")

    fig2 = plt.figure(figsize=(8, 10))
    gs2 = GridSpec(4, 3, figure=fig2)
    ax21 = fig2.add_subplot(gs2[0, :])
    ax22 = fig2.add_subplot(gs2[1, :])
    ax23 = fig2.add_subplot(gs2[2, :])
    ax24 = fig2.add_subplot(gs2[3, 0])
    ax25 = fig2.add_subplot(gs2[3, 1])
    ax26 = fig2.add_subplot(gs2[3, 2])

    for vi in np.arange(0, np.shape(data)[0]):
        plume = diagnostics.plume(
            coords[vi], data[vi], ["flx", "ave", "min", "max"], sec=False
        )
        d = plume["d"] / 1000
        taw = np.nanmax(data[vi]["tref"])

        melt = np.abs(SHIflx[vi]["fwfx"]) / 1000 * dx
        cmelt = np.nancumsum(melt)

        color = colors(exp[vi] / np.max(exp))
        line = lines[geo[vi]]
        marker = markers[geo[vi]]

        u_ave = plume["u_ave"]
        t_ave = plume["t_ave"]
        s_ave = plume["s_ave"]
        s_ave[np.argwhere(np.isnan(s_ave))] = 0

        s_fil = uniform_filter1d(s_ave, size=20)
        print(s_ave)
        print(s_fil)

        u_max = plume["u_max"]
        t_min = plume["t_min"]
        s_min = plume["s_min"]
        flx = plume["flx"]

        ax12.plot(coords[0]["x"] / 1000, plume["thick"], color="black")
        ax13.plot(d, t_ave, color=color, linestyle=line, alpha=0.5, label=path[vi])
        ax14.plot(d, s_ave, color=color, linestyle=line, alpha=0.1, label=path[vi])
        ax14.plot(d, s_fil, color=color, linestyle=line, alpha=0.5, label=path[vi])
        ax15.plot(d, u_max, color=color, linestyle=line, alpha=0.5, label=path[vi])

        # ------------Figure 2-------------

        ax21.plot(d, flx, color=color, linestyle=line, alpha=0.5, label=path[vi])
        ax22.plot(d, melt, color=color, linestyle=line, alpha=0.5)
        ax23.plot(
            d,
            np.cumsum(-SHIflx[vi]["fwfx"] / 1000) / flx,
            color=color,
            linestyle=line,
            alpha=0.5,
        )

        ax24.plot(taw, np.nanmax(flx), marker, color=color)
        ax25.plot(taw, cmelt[np.nanargmax(flx)], marker, color=color, label=path[vi])
        ax26.plot(taw, cmelt[np.nanargmax(flx)] / np.nanmax(flx), marker, color=color)

    # ax11.set_xticklabels([])
    ax11.set_ylabel("Ice depth")
    ax11.grid("both")
    ax11.set_xlim(0, 20.5)

    ax13.set_xticklabels([])
    ax13.set_ylabel("ave plume T")
    ax13.grid("both")
    ax13.set_xlim(0, 20.5)

    ax14.set_xticklabels([])
    ax14.set_ylabel("ave plume S")
    ax14.grid("both")
    ax14.set_xlim(0, 20.5)
    ax14.set_ylim(0, 35.1)

    ax15.set_ylabel("ave plume U")
    ax15.set_ylabel("distance along Ice")
    ax15.grid("both")
    ax15.set_xlim(0, 20.5)

    # ------------Figure 2-------------
    ax21.set_xticklabels([])
    ax21.set_ylabel("Plume Flux [m^2/s]")
    ax21.grid("both")
    ax21.set_xlim(0, 20.5)

    ax22.set_ylabel("melt from ice")
    ax22.set_xticklabels([])
    ax22.grid("both")
    ax22.set_xlim(0, 20.5)

    ax23.set_xlabel("dist along ice")
    ax23.set_ylabel("cum(melt)/flx")
    ax23.grid("both")
    ax23.set_xlim(0, 20.5)
    ax23.set_ylim(0, 0.001)

    ax24.set_ylabel("Plume Vol. Flux (QP) [m^2/s]")
    ax24.set_xlabel("AW Temperature")
    ax24.grid("both")

    ax25.set_ylabel("Cummulative Melt (M)[m^2/s]")
    ax25.set_xlabel("AW Temperature")
    ax25.grid("both")

    ax26.set_ylabel("sum(M)/max(QP)")
    ax26.set_xlabel("AW Temperature")
    ax26.grid("both")
    ax26.ticklabel_format(axis="y", style="sci", scilimits=[-1, 2])

    fig1.tight_layout()
    fig2.tight_layout()

    ax15.legend(loc="center right", ncol=1)  # , bbox_to_anchor=(-0.00, 0))
    ax21.legend(loc="center right", ncol=1)  # , bbox_to_anchor=(-0.00, 0))
    fig1.savefig("plots/tsu" + figname, facecolor="white")
    fig2.savefig("plots/flux" + figname, facecolor="white")


def plot_ekin(figname, path, flx=[], prx=[]):

    pri = np.zeros(np.shape(prx)[0]).astype("int")
    lines, markers = load_lines()
    colors = load_colors()
    lines = np.array(lines)

    exp, geo = get_expgeo(path)

    [fig, axs] = plt.subplots(3, 1, figsize=(16, 9))
    for vi in np.arange(0, np.shape(path)[0]):

        load_all(path[vi], "times")
        time = data["time"] / 86400
        upr = np.zeros([np.shape(data["u_all"])[0], coords["nz"]])
        print("loaded")

        color = colors(exp[vi] / np.max(exp))
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
        print(np.shape(motQ))

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

    axs[0].set_xlabel("Model Days")
    axs[0].set_ylabel("Kinetic Energy")
    axs[0].grid("both")
    axs[0].legend(ncol=3, loc=8)

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
