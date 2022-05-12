from turtle import width
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from scipy import interpolate
import pickle
import csv
from scipy.ndimage.filters import uniform_filter1d

# import gsw
import importlib
from ordered_set import OrderedSet

from . import loadMIT
from . import toolsMIT as TM
from . import diagnostics

importlib.reload(diagnostics)


def plot_sec(figname, path, vi, prx):
    tAlpha = (-0.4e-4,)
    sBeta = (8.0e-4,)
    rnil = 999.8
    data = TM.data
    coords = TM.coords
    SHIflx = TM.SHIflx

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

    psi = np.flipud(np.nancumsum(np.flip(u, axis=0) * dz, axis=0))
    r = rnil * (1 + tAlpha * t + sBeta * s) - 1000

    rgrid = np.linspace(27.7, np.nanmax(r), 101)
    psi_r = np.zeros([np.shape(rgrid)[0], np.shape(x)[0]])

    for i in np.arange(0, len(x)):
        for ri in np.arange(0, len(rgrid)):
            ind = np.nonzero(r[:, i] > rgrid[ri])[0]
            if np.any(ind):
                psi_r[ri, i] = np.nansum(np.flip(u[ind, i], axis=0), axis=0) * -dz

    dep = coords[vi]["topo"]

    if len(dep) > len(x):
        dep = dep[: len(x)]
        ice = ice[: len(x)]

    [fig, axs] = plt.subplots(2, 1, figsize=(8, 8))

    if prx is not None:
        for i in np.arange(0, np.shape(prx)[0]):
            axs[0].plot(np.ones(coords[vi]["nz"]) * prx[i] / 1000, z / 1000, "--k")

    cf0 = axs[0].contourf(
        x / 1000,
        z / 1000,
        r,
        np.linspace(27.73, 27.78, 21),
        cmap="viridis_r",
        extend="both",
    )

    axs[0].contour(
        x / 1000,
        z / 1000,
        psi,
        np.linspace(1, 10, 10),
        colors="white",
        alpha=0.5,
    )
    # axs[0].plot(x / 1000, dep / 1000, "k", linewidth=2)
    # axs[0].plot(x[icei] / 1000, ice[icei] / 1000, "k", linewidth=2)
    axs[0].set_title("Exp: {} - Day {}-{}".format(path[vi], time[0], time[-1]))
    axs[0].set_xlabel("Distance from grounding line [km]")
    axs[0].set_ylabel("depth in [km]")

    if SHIflx[i] == 0:
        cbar1 = fig.colorbar(
            cf0, ax=axs[0], orientation="horizontal", fraction=0.1, anchor=(1.0, 0.1)
        )
    else:
        fwf = SHIflx[vi]["fwfx"]

        fwf[ice > 0] = np.nan
        meltflux = np.nansum(fwf) / 1000 * dx
        melt = np.nanmean(fwf) / 1000 * 86400 * 365
        axs[0].text(
            1,
            -0.1,
            r"Melt flux: ${:7.3f} m^2/s$".format(meltflux),
            fontsize=12,
        )
        axs[0].text(
            1,
            -0.2,
            r"Ave melt rate: ${:5.1f} m/yr$".format(melt),
            fontsize=12,
        )
        ax02 = axs[0].twinx()
        ax02.plot(x / 1000, fwf, color="green")
        ax02.set_ylabel("fresh water flux")
        ax02.tick_params(axis="y", colors="green")
        ax02.spines["right"].set_color("green")
        ax02.ticklabel_format(axis="y", style="sci", scilimits=[-1, 2])
        ax02.set_xlim(0, 30)
        ax02.set_ylim(2 * np.nanmin(fwf), 0)
        cbar1 = fig.colorbar(
            cf0, ax=ax02, orientation="horizontal", fraction=0.08
        )  # , anchor=(1.0, 0.1)

    cf1 = axs[1].contourf(
        x / 1000,
        rgrid,
        -psi_r,
        np.linspace(0, 10, 11),
        cmap="Reds",
        extend="both",
    )
    axs[1].set_xlabel("Distance from grounding line [km]")
    axs[1].set_ylabel("Sigma")
    axs[1].set_xlim([0, 30])
    axs[1].set_ylim([27.73, 27.78])
    axs[1].invert_yaxis()
    cbar2 = fig.colorbar(cf1, ax=axs[1], orientation="horizontal", fraction=0.08)

    cbar1.set_label("Sigma")
    cbar2.set_label("Transport Stream function")
    plt.tight_layout()
    fig.savefig("plots/" + path[vi] + figname, facecolor="white", dpi=600)
    plt.show()
    plt.close("all")


def plot_prof(fig_name, path, prx, Gade=False, SalT=True):

    data = TM.data
    coords = TM.coords
    t, s, u, w = TM.init_uts(data)
    upr, spr, tpr = TM.init_prof(coords[0], data, prx)
    dels = np.zeros(np.shape(upr))

    Q = np.zeros([np.shape(data)[0], np.shape(prx)[0], coords[0]["nz"]])
    mot = np.zeros([np.shape(data)[0], np.shape(prx)[0]])
    maxi = np.zeros([np.shape(data)[0], np.shape(prx)[0]])
    pri = np.zeros([np.shape(prx)[0]])
    sref = data[0]["sref"]
    dz = -coords[0]["dz"][0]
    dy = coords[0]["dx"][0]

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

    ds = 0.01
    s = np.arange(34.7, 35 + ds, ds)
    strans = np.zeros([np.shape(path)[0], np.shape(s)[0]])
    # sflx = (
    #     np.nansum(dels * upr, axis=2)
    #     / np.shape(upr)[2]
    #     / np.max(sref)
    #     * 3600
    #     * 24
    #     * 365
    # )
    sflx = np.nansum(dels * upr, axis=2) / np.max(sref) * dz * dy
    tAlpha = (-0.4e-4,)
    sBeta = (8.0e-4,)
    mot = np.array(mot)
    maxi = np.array(maxi)
    Q = np.array(Q)
    upr = np.array(upr)
    tpr = np.array(tpr)
    spr = np.array(spr)

    fig = plt.figure(figsize=(8, 8))
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
    axs[0].set_xlabel("Temperature")

    # axs[1].plot(var[0]["sref"], coords["z"] / 1000, "--k", label="sref")
    axs[1].grid("both")
    axs[1].set_ylim(-1.050, 0.050)
    axs[1].set_xlim(34.9, 35)
    axs[1].set_xlabel("Salinity")

    axs[2].grid("both")
    axs[2].set_ylim(-1.050, 0.050)
    axs[2].set_xlim(27.490, 27.91)
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
    axs[5].set_ylabel("Volume transport [$m^2/s$]")
    axs[5].set_xlabel("Salinity")
    axs[5].set_xticks(s[::2])
    axs[5].set_xlim(34.87, 35.01)

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
        rmax = rpr[np.nanargmax(upr[vi, i, :])]
        z = TM.coords[vi]["z"]
        zmax = TM.coords[vi]["z"][np.nanargmax(upr[vi, i, :])]
        tref = TM.data[vi]["tref"]

        color, line, marker = TM.identify(path[vi], tref[-1])

        soff = ((tref[-1] + 2.5) / 10) * ds * 0.9

        rpref = 999.8 * (1 + tAlpha * tref + sBeta * sref) - 1000
        n = 0
        for ss in s:
            ind = np.where(
                (spr[vi, i, :] > ss - ds / 2) & (spr[vi, i, :] < ss + ds / 2)
            )[0]
            if np.any(ind):
                strans[vi, n] = np.nansum(upr[vi, i, ind]) / np.nansum(
                    np.abs(upr[vi, i, :])
                )
            n += 1

        if Gade:
            TG, SG = diagnostics.gadeline(tpr[vi, i, :], spr[vi, i, :], z)
            axs[3].plot(SG, TG, "--k", label="Gade ")

        for i in np.arange(0, np.shape(prx)[0]):
            axs[0].plot(
                tref,
                z / 1000,
                ":",
                color=color,
            )
            axs[0].plot(
                tpr[vi, i, :],  # - tref,
                z / 1000,
                line,
                color=color,
            )
            axs[1].plot(
                sref,
                z / 1000,
                "k:",
                color=color,
            )
            axs[1].plot(
                spr[vi, i, :],  # - sref,
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
                ":",
                color=color,
            )
            axs[2].plot(
                rmax,
                zmax / 1000,
                marker,
                color=color,
            )
            axs[3].plot(
                upr[vi, i, :],
                z / 1000,
                line,
                color=color,
            )
            axs[4].plot(
                tref[-1],
                -sflx[vi, i],
                marker,
                color=color,
            )
            axs[5].bar(s + soff, strans[vi, :], color=color, width=0.0007)
        # axs[5].plot(
        #     spr[vi, i, :],
        #     upr[vi, i, :],
        #     line,
        #     color=color,
        # )

    plt.tight_layout()
    axs[1].legend(loc="lower center", bbox_to_anchor=(0.1, 0.0), ncol=2)
    fig.savefig("plots/" + fig_name, facecolor="white", dpi=600)
    fig.show()


def plot_plume(figname, path, sec=False, which=["tsu", "flux", "buoy", "sum"]):
    importlib.reload(diagnostics)

    coords = TM.coords
    data = TM.data
    SHIflx = TM.SHIflx

    lam1 = -5.75e-2
    lam2 = 9.01e-2
    lam3 = -7.61e-4
    tAlpha = -0.4e-4
    sBeta = 8.0e-4
    rnil = 999.8
    g = -9.81
    z = coords[0]["z"]
    dx = coords[0]["dx"][0]

    if "tsu" in which:
        heights = [3, 3, 3, 3]
        fig1 = plt.figure(figsize=(8, 8))
        gs1 = GridSpec(4, 1, figure=fig1, height_ratios=heights)
        ax11 = fig1.add_subplot(gs1[0, :])
        ax12 = fig1.add_subplot(gs1[1, :])
        ax13 = fig1.add_subplot(gs1[2, :])
        ax14 = fig1.add_subplot(gs1[3, :])

        ax11.set_ylabel("Plume \nthickness")
        ax11.grid("both")
        ax11.set_xlim(0, 15)

        ax12.set_xticklabels([])
        ax12.set_ylabel("$S_{BL}-S_{AW}$")
        ax12.grid("both")
        ax12.set_xlim(0, 15)
        ax12.set_ylim(-0.21, 0.01)
        ax12.ticklabel_format(axis="y", style="sci", scilimits=[-2, 2])

        ax13.set_xticklabels([])
        ax13.set_ylabel("$T_{BL}-T_{AW}$")
        ax13.grid("both")
        ax13.set_ylim(-1.4, 0.01)
        ax13.set_xlim(0, 15)

        ax14.set_ylabel("BL veloctiy")
        ax14.set_xlabel("distance along Ice")
        ax14.grid("both")
        ax14.set_xlim(0, 15)

    if "flux" in which:
        fig2 = plt.figure(figsize=(8, 6))
        gs2 = GridSpec(3, 3, figure=fig2)
        ax21 = fig2.add_subplot(gs2[0, :])
        ax22 = fig2.add_subplot(gs2[1, :])
        ax23 = fig2.add_subplot(gs2[2, 0])
        ax24 = fig2.add_subplot(gs2[2, 1])
        ax25 = fig2.add_subplot(gs2[2, 2])

        ax21.set_xticklabels([])
        ax21.set_ylabel("Plume Flux [m^2/s]")
        ax21.grid("both")
        ax21.set_xlim(0, 15)

        ax22.set_ylabel("melt from ice")
        ax22.set_xlabel("dist along ice")
        ax22.set_xticklabels([])
        ax22.grid("both")
        ax22.set_xlim(0, 15)

        ax23.set_ylabel("Plume Vol. Flux \n(QP) [m^2/s]")
        ax23.set_xlabel("AW Temperature")
        ax23.grid("both")

        ax24.set_ylabel("Cummulative Melt \n(M)[m^2/s]")
        ax24.set_xlabel("AW Temperature")
        ax24.grid("both")

        ax25.set_ylabel("sum(M)/max(QP)")
        ax25.set_xlabel("AW Temperature")
        ax25.grid("both")
        ax25.ticklabel_format(axis="y", style="sci", scilimits=[-1, 2])

    if "buoy" in which:
        fig3 = plt.figure(figsize=(8, 6))
        gs3 = GridSpec(2, 3, figure=fig3)
        ax31 = fig3.add_subplot(gs3[0, :])
        ax32 = fig3.add_subplot(gs3[1, 0])
        ax33 = fig3.add_subplot(gs3[1, 1])
        ax34 = fig3.add_subplot(gs3[1, 2])

        ax31.set_xticklabels([])
        ax31.set_ylabel("Plume Buoyancy\ntotal")
        ax31.grid("both")
        ax31.set_xlim(0, 15)
        ax31.set_ylim(-0.25, 0.51)
        ax31.ticklabel_format(axis="y", style="sci", scilimits=[-1, 2])
        ax31.legend()

        ax32.set_ylabel("Plume Buoyancy\ntotal")
        ax32.set_ylim(-0.01, 0.51)
        ax32.grid("both")

        ax33.set_ylabel("Sal")
        ax33.set_ylim(-0.01, 0.51)
        ax33.grid("both")

        ax34.set_ylabel("Temp")
        ax34.set_ylim(-0.61, 0.01)
        ax34.grid("both")

    if "sum" in which:
        fig4 = plt.figure(figsize=(8, 6))
        gs4 = GridSpec(2, 3, figure=fig4)
        ax41 = fig4.add_subplot(gs4[0, 0])
        ax42 = fig4.add_subplot(gs4[0, 1])
        ax43 = fig4.add_subplot(gs4[0, 2])
        ax44 = fig4.add_subplot(gs4[1, 0])
        ax45 = fig4.add_subplot(gs4[1, 1])
        ax46 = fig4.add_subplot(gs4[1, 2])

        ax41.set_title("Q [m^2/s]")
        ax41.set_xlim(-2.51, 6.01)
        ax41.grid("both")
        ax41.set_xticklabels([])
        #
        ax42.set_title("M[m^2/s]")
        # ax42.set_ylim(0, 0.075)
        ax42.set_xlim(-2.51, 6.01)
        ax42.grid("both")
        ax42.set_xticklabels([])
        #
        ax43.set_title("M/QP")
        ax43.set_ylim(0, 5e-3)
        ax43.set_xlim(-2.51, 6.01)
        ax43.grid("both")
        ax43.set_xticklabels([])
        ax43.ticklabel_format(axis="y", style="sci", scilimits=[-1, 2])

        ax44.set_title("$b_{total}$")
        ax44.set_ylabel("Plume Buoyancy")
        ax44.set_xlabel("AW Temperature")
        ax44.set_ylim(-0.01, 0.61)
        ax44.set_xlim(-2.51, 6.01)
        ax44.grid("both")

        ax45.set_title("$b_{Sal}$")
        ax45.set_xlabel("AW Temperature")
        ax45.set_ylim(-0.01, 0.61)
        ax45.set_xlim(-2.51, 6.01)
        ax45.grid("both")

        ax46.set_title("$b_{Temp}$")
        ax46.set_xlabel("AW Temperature")
        ax46.set_ylim(-0.61, 0.01)
        ax46.set_xlim(-2.51, 6.01)
        ax46.grid("both")

    if "select1" in which:
        fig5 = plt.figure(figsize=(8, 4))
        gs5 = GridSpec(2, 1, figure=fig5)
        ax51 = fig5.add_subplot(gs5[0, 0])
        ax52 = fig5.add_subplot(gs5[1, 0])
        ax51.set_xticklabels([])
        ax51.set_ylabel("Plume \nthickness")
        ax51.grid("both")
        ax51.set_xlim(0, 15)
        ax51.set_ylim(-2, 132)

        ax52.set_xticklabels([])
        ax52.set_ylabel("BL veloctiy")
        ax52.grid("both")
        ax52.set_xlim(0, 15)
        ax52.set_ylim(-0.01, 0.17)

    if "select2" in which:
        fig6 = plt.figure(figsize=(9, 3))
        gs6 = GridSpec(1, 3, figure=fig6)
        ax61 = fig6.add_subplot(gs6[0, 0])
        ax62 = fig6.add_subplot(gs6[0, 1])
        ax63 = fig6.add_subplot(gs6[0, 2])

        ax61.set_title("Q [m^2/s]")
        ax61.grid("both")
        ax61.set_xlabel("AW Temperature")

        ax62.set_title("M[m^2/s]")
        ax62.grid("both")
        ax62.set_xlabel("AW Temperature")

        ax63.set_title("$TF = T_{BL} - T_{ice} [deg C]$")
        ax63.grid("both")
        ax63.set_xlabel("AW Temperature")

    if "nondim" in which:
        fig7 = plt.figure(figsize=(8, 4))
        gs7 = GridSpec(2, 1, figure=fig7)
        ax71 = fig7.add_subplot(gs7[0, 0])
        ax72 = fig7.add_subplot(gs7[1, 0])

        ax71.grid("both")
        ax71.set_ylabel("Froude Number")
        ax71.set_xticklabels([])
        ax71.set_ylim(0, 2)
        ax71.set_xlim(0, 15)

        ax72.grid("both")
        ax72.set_ylabel("Richardson Number")
        ax72.set_xlabel("distance along Ice")
        ax72.set_ylim(0, 2)
        ax72.set_xlim(0, 15)

    for vi in np.arange(0, np.shape(data)[0]):
        plume = diagnostics.plume(coords[vi], data[vi], ["flx"])

        if sec:
            fig = plt.figure(figsize=(12, 15))
            ax1 = plt.subplot(3, 1, 1)
            ax2 = plt.subplot(3, 1, 2)
            ax3 = plt.subplot(3, 1, 3)
            axs = [ax1, ax2, ax3]

            ax1.contourf(plume["d"], plume["z_plu"], plume["t_plu"])
            ax2.contourf(plume["d"], plume["z_plu"], plume["u_plu"])
            ax3.contourf(plume["d"], plume["z_plu"], plume["w_plu"])

            ax1.set_xticklabels([])
            ax1.set_ylabel(" plume T")
            ax1.grid("both")
            ax1.set_xlim(0, 7.5e3)

            ax2.set_xticklabels([])
            ax2.set_ylabel(" plume U")
            ax2.grid("both")
            ax2.set_xlim(0, 7.5e3)

            ax3.set_ylabel(" plume W")
            ax3.grid("both")
            ax3.set_xlim(0, 7.5e3)
            plt.show()

        x = np.squeeze(plume["d"] / 1000)
        d = plume["thick"]
        taw = data[vi]["tref"][-1]
        ice = coords[vi]["ice"]
        if len(ice) > len(x):
            ice = ice[: len(x)]
        tref = data[vi]["tref"][-1]
        sref = data[vi]["sref"][-1]
        t0 = 1
        s0 = 35

        melt = np.abs(SHIflx[vi]["fwfx"]) / 1000 * dx
        cmelt = np.nancumsum(melt)

        color, line, marker = TM.identify(path[vi], tref)

        u_ice = plume["u_plu"][0, :]
        w_ice = plume["w_plu"][1, :]
        vel_ice = np.sqrt(u_ice**2 + w_ice**2)

        t_ice = plume["t_plu"][0, :]
        t_ave = np.nanmean(plume["t_plu"], axis=0)
        s_ice = plume["s_plu"][0, :]
        s_ave = np.nanmean(plume["s_plu"], axis=0)
        vel_plu = np.sqrt(plume["u_plu"] ** 2 + plume["w_plu"] ** 2)
        vel_ave = np.nanmean(vel_plu, axis=0)

        nans, i = diagnostics.nan_helper(vel_ice)
        vel_ice[nans] = np.interp(i(nans), i(~nans), vel_ice[~nans])
        nans, i = diagnostics.nan_helper(vel_ave)
        vel_ave[nans] = np.interp(i(nans), i(~nans), vel_ave[~nans])
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
        va_fil = uniform_filter1d(vel_ave, size=20)
        ti_fil = uniform_filter1d(t_ice, size=20)
        ta_fil = uniform_filter1d(t_ave, size=20)
        d_fil = uniform_filter1d(d, size=20)

        t_f = lam1 * si_fil + lam2 + ice * lam3
        TF = ti_fil - t_f

        rho1 = (1 + (sref - s0) * sBeta + (tref - t0) * tAlpha) * rnil
        rho2 = (1 + (sa_fil - s0) * sBeta + (ta_fil - t0) * tAlpha) * rnil

        rho2s = (1 + (sa_fil - s0) * sBeta) * rnil
        rho2t = (1 + (ta_fil - t0) * tAlpha) * rnil

        s_buo = (rho1 - rho2s) * g
        t_buo = (rho1 - rho2t) * g

        g_red = (rho1 - rho2) / rho1 * -g

        if vi == 0:
            s_b_ref = (sa_fil - sref) * 8e-4 * g
            t_b_ref = (ta_fil - tref) * -0.4e-4 * g
            t_force = t_ice - t_f

        flx = plume["flx"]

        fr = va_fil / np.sqrt(g_red * d_fil)
        ri = g_red * d_fil / va_fil**2

        if "tsu" in which:
            ax11.plot(x, d_fil, color=color)
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
            ax14.plot(
                x, vel_fil, color=color, linestyle=line, alpha=0.7, label=path[vi]
            )
            # ax14.plot(
            #     x, vel_ice, color=color, linestyle=line, alpha=0.1, label="_nolegend_"
            # )

        # ------------Figure 2-------------
        if "flux" in which:
            ax21.plot(x, flx, color=color, linestyle=line, alpha=0.7, label=path[vi])
            ax22.plot(
                x,
                np.nancumsum(melt),
                color=color,
                linestyle=line,
                alpha=0.7,
                label=path[vi],
            )

            ax23.plot(taw, np.nanmax(flx), marker, color=color)
            ax24.plot(
                taw, cmelt[np.nanargmax(flx)], marker, color=color, label=path[vi]
            )
            ax25.plot(
                taw, cmelt[np.nanargmax(flx)] / np.nanmax(flx), marker, color=color
            )

        # ------------Figure 3-------------
        if "buoy" in which:
            if vi == 0:
                tlabel = "Temp"
                slabel = "Sal"
                label = "Total"
                ax31.plot(
                    x, s_buo + 10, color="k", linestyle="-", alpha=0.7, label=label
                )
                ax31.plot(
                    x, s_buo + 10, color="k", linestyle="--", alpha=0.7, label=tlabel
                )
                ax31.plot(
                    x, s_buo + 10, color="k", linestyle=":", alpha=0.7, label=slabel
                )

            else:
                tlabel = "_nolegend"
                slabel = "_nolegend"
                label = "_nolegend"

            ax31.plot(x, s_buo + t_buo, color=color, linestyle=line, alpha=0.7)
            ax31.plot(x, s_buo, color=color, linestyle=":", alpha=0.7)
            ax31.plot(x, t_buo, color=color, linestyle="--", alpha=0.7)
            ax32.plot(
                tref,
                np.nanmean(t_buo[0 : np.nanargmax(flx)] + s_buo[0 : np.nanargmax(flx)]),
                marker,
                color=color,
            )
            ax33.plot(
                tref,
                np.nanmean(s_buo[0 : np.nanargmax(flx)]),
                marker,
                color=color,
            )
            ax34.plot(
                tref,
                np.nanmean(t_buo[0 : np.nanargmax(flx)]),
                marker,
                color=color,
                label=path[vi],
            )

        # ------------Figure 4-------------
        if "sum" in which:
            ax41.plot(taw, np.nanmax(flx), marker, color=color)
            ax42.plot(taw, cmelt[np.nanargmax(flx)], marker, color=color)
            ax43.plot(
                taw, cmelt[np.nanargmax(flx)] / np.nanmax(flx), marker, color=color
            )
            ax44.plot(
                tref,
                np.nanmean(t_buo[0 : np.nanargmax(flx)] + s_buo[0 : np.nanargmax(flx)]),
                marker,
                color=color,
            )
            ax45.plot(
                tref,
                np.nanmean(s_buo[0 : np.nanargmax(flx)]),
                marker,
                color=color,
            )
            ax46.plot(
                tref,
                np.nanmean(t_buo[0 : np.nanargmax(flx)]),
                marker,
                color=color,
            )

        # ------------Figure 5-------------
        if "select1" in which:
            ax51.plot(x, d_fil, color=color, linestyle=line, label=path[vi])
            ax52.plot(x, vel_fil, color=color, linestyle=line, label=path[vi])

        # ------------Figure 6-------------
        if "select2" in which:
            ax61.plot(taw, np.nanmax(flx), marker, color=color, label=path[vi])
            ax62.plot(taw, cmelt[np.nanargmax(flx)], marker, color=color)
            ax63.plot(taw, np.nanmean(TF[0 : np.nanargmax(flx)]), marker, color=color)
            ax63.plot(
                taw, np.nanmin(TF[0 : np.nanargmax(flx)]), marker="^", color=color
            )
            ax63.plot(
                taw, np.nanmax(TF[0 : np.nanargmax(flx)]), marker="v", color=color
            )

        # ------------Figure 6-------------
        if "nondim" in which:
            ax71.plot(x, fr, color=color, linestyle=line, label=path[vi])
            ax72.plot(x, ri, color=color, linestyle=line, label=path[vi])

        f = open("fitting_data{}.csv".format(figname), "a")
        writer = csv.writer(f)
        writer.writerow(
            [
                taw,
                np.nanmax(flx),
                cmelt[np.nanargmax(flx)],
                cmelt[np.nanargmax(flx)] / np.nanmax(flx),
            ]
        )
        f.close()

    if "tsu" in which:
        fig1.tight_layout()
        ax14.legend(loc="center right", ncol=1, bbox_to_anchor=(1.00, 1.5))
        fig1.savefig("plots/tsu" + figname, facecolor="white")

    if "flux" in which:
        fig2.tight_layout()
        ax22.legend(loc="center right", ncol=1, bbox_to_anchor=(1.00, 1))
        fig2.savefig("plots/flux" + figname, facecolor="white")

    if "buoy" in which:
        fig3.tight_layout()
        ax34.legend(loc="lower center", ncol=3, bbox_to_anchor=(-0, 0))
        fig3.savefig("plots/buoy" + figname, facecolor="white")

    if "sum" in which:
        fig4.tight_layout()
        # ax46.legend(loc="lower center", ncol=3, bbox_to_anchor=(-0, 0))
        fig4.savefig("plots/sum" + figname, facecolor="white")

    if "select1" in which:
        fig5.tight_layout()
        ax51.legend(loc="upper left", ncol=4)  # , bbox_to_anchor=(0.5, 0))
        fig5.savefig("plots/DTU" + figname, facecolor="white", dpi=600)

    if "select2" in which:
        fig6.tight_layout()
        # ax61.legend(loc="lower center", ncol=1, bbox_to_anchor=(0.5, 0))
        fig6.savefig("plots/Melt" + figname, facecolor="white", dpi=600)

    if "nondim" in which:
        fig7.tight_layout()
        ax71.legend(loc="lower center", ncol=4)
        fig7.savefig("plots/nondim" + figname, facecolor="white", dpi=600)


def plot_ekin(figname, path, flx=[], prx=[]):

    pri = np.zeros(np.shape(prx)[0]).astype("int")

    [fig, axs] = plt.subplots(3, 1, figsize=(8, 5))

    print("start plotting {} timeseries".format(np.shape(path)[0]))
    for vi in np.arange(0, np.shape(path)[0]):
        print("plot {} of {}".format(vi + 1, np.shape(path)[0]))
        data, coords, SHIflx, gammas = TM.load_single(path[vi], "times")

        time = data["time"] / 86400
        step = np.squeeze(np.arange(0, np.shape(time)[0]) + 1)
        upr = np.zeros([np.shape(data["u_all"])[0], coords["nz"]])

        color, line, marker = TM.identify(path[vi], data["tref"][-1])
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

        if len(path[vi]) < 12:
            label = "Control"
        else:
            label = path[vi][12:]

        axs[0].plot(
            step,
            ekin,
            line,
            color=color,
            label=label,
        )

        axs[1].plot(
            step,
            melt,
            line,
            color=color,
        )

        axs[2].plot(
            step,
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

    fig = plt.figure(figsize=(12, 10))
    ax1 = plt.subplot(2, 3, 1)
    ax2 = plt.subplot(2, 3, 2)
    ax3 = plt.subplot(2, 3, 3)
    ax4 = plt.subplot(2, 1, 2)
    axs = [ax1, ax2, ax3, ax4]
    ax0 = axs[0].twiny()
    ax1 = axs[1].twiny()
    for vi in np.arange(0, np.shape(var)[0]):

        color, line, marker = TM.identify(path)

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

    fig = plt.figure(figsize=(12, 10))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    axs = [ax1, ax2]

    for vi in np.arange(0, np.shape(var)[0]):

        color, line, marker = TM.identify(path)

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


def plot_evox(figname, path, prx):
    data = TM.data
    coords = TM.coords
    t, s, u, w = TM.init_uts(data)
    upr, spr, tpr = TM.init_prof(coords[0], data, prx)
    colors = cm.get_cmap("cividis")

    for vi in np.arange(0, np.shape(path)[0]):
        z = coords[vi]["z"]
        t[vi, :, :], s[vi, :, :], u[vi, :, :], w[vi, :, :] = loadMIT.ave_tsu(
            data[vi]["t_all"], data[vi]["s_all"], data[vi]["u_all"], data[vi]["w_all"]
        )
        t = np.array(t)
        s = np.array(s)
        u = np.array(u)

        tref = TM.data[vi]["tref"]
        sref = TM.data[vi]["sref"]

        fig = plt.figure(figsize=(8, 4))
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)

        ax1.plot(tref, z / 1000, "--k")
        ax2.plot(sref, z / 1000, "--k")

        for i in np.arange(0, len(prx)):
            label = "at {} km".format(prx[i] / 1000)
            pri = np.min(np.where(coords[vi]["x"] >= prx[i])[0])
            tpr[vi, i, :] = t[vi, :, pri]
            spr[vi, i, :] = s[vi, :, pri]
            color = colors(prx[i] / np.max(prx))
            ax1.plot(tpr[vi, i, :], z / 1000, color=color, label=label)
            ax2.plot(spr[vi, i, :], z / 1000, color=color, label=label)

        # axs[0].plot(var[0]["tref"], coords["z"] / 1000, "--k", label="tref")
        ax1.grid("both")
        ax1.set_ylim(-1.050, 0.050)
        ax1.set_xlim(-0.21, 0.41)
        ax1.set_ylabel("depth [km]")
        ax1.set_xlabel("Temperature")

        # axs[1].plot(var[0]["sref"], coords["z"] / 1000, "--k", label="sref")
        ax2.grid("both")
        ax2.set_ylim(-1.050, 0.050)
        ax2.set_xlim(34.89, 35.01)
        ax2.set_xlabel("Salinity")

        plt.tight_layout()
        ax2.legend(loc="upper center", bbox_to_anchor=(-0.1, 1), ncol=4)
        fig.savefig("plots/{}_{}".format(path[vi], figname), facecolor="white")
