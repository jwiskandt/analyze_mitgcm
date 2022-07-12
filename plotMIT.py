from gettext import dgettext
from importlib import metadata
from turtle import fillcolor, width
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from scipy import interpolate
import pickle
import csv
from scipy.ndimage.filters import uniform_filter1d
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence
import pandas as pd

# import gsw
import importlib
from ordered_set import OrderedSet

from . import loadMIT
from . import toolsMIT as TM
from . import diagnostics

importlib.reload(diagnostics)


def plot_sec(path, vi, prx=[25e3], which=["ave"]):
    tAlpha = (-0.4e-4,)
    sBeta = (8.0e-4,)
    rnil = 999.8
    data = TM.data
    sref = data[vi]["sref"]
    tref = data[vi]["tref"]
    saw = sref[-1]
    taw = tref[-1]
    coords = TM.coords
    SHIflx = TM.SHIflx
    x = coords[vi]["x"]
    pri = np.where(x > prx)[0][0]
    z = coords[vi]["z"]
    ice = coords[vi]["ice"]
    hfac = coords[vi]["hfac"]
    dx = coords[vi]["dx"][0]
    dy = coords[vi]["dx"][0]
    dz = -coords[vi]["dz"][0]
    time = data[vi]["time"] / 86400
    if np.nanmax(time) > 100:
        time = time / 2
    dep = coords[vi]["topo"]

    if len(dep) > len(x):
        dep = dep[: len(x)]
        ice = ice[: len(x)]

    if "ave" in which:
        figname = "_sec.png"

        (t, s, u, w) = loadMIT.ave_tsu(
            data[vi]["t_all"], data[vi]["s_all"], data[vi]["u_all"], data[vi]["w_all"]
        )
        ds = np.zeros(np.shape(s))
        for i in np.arange(0, np.shape(s)[1]):
            ds[:, i] = s[:, i] - s[:, pri]

        # nan above the ice
        for i in np.arange(1, coords[vi]["nx"]):
            t[z > ice[i], i] = float("nan")
            s[z > ice[i], i] = float("nan")
            u[z > ice[i], i] = float("nan")
            w[z > ice[i], i] = float("nan")

        psi = np.flipud(np.nancumsum(np.flip(u, axis=0) * dz, axis=0))
        r = rnil * (1 + tAlpha * t + sBeta * s) - 1000

        rgrid = np.linspace(np.nanmin(r), np.nanmax(r), 101)
        sgrid = np.linspace(34.9, np.nanmax(s), 21)
        psi_r = np.zeros([np.shape(rgrid)[0], np.shape(x)[0]])
        psi_s = np.zeros([np.shape(sgrid)[0], np.shape(x)[0]])

        for i in np.arange(0, len(x)):
            for ri in np.arange(0, len(rgrid)):
                ind = np.nonzero(r[:, i] > rgrid[ri])[0]
                if np.any(ind):
                    psi_r[ri, i] = np.nansum(np.flip(u[ind, i], axis=0), axis=0) * -dz
            for si in np.arange(0, len(sgrid) - 1):
                sind = np.nonzero((s[:, i] >= sgrid[si]) & (s[:, i] < sgrid[si + 1]))[0]
                if np.any(sind):
                    psi_s[si, i] = np.nansum(np.flip(u[sind, i], axis=0), axis=0) * -dz

        print(np.nanmax(psi_s))

        [fig, axs] = plt.subplots(1, 1, figsize=(8, 8))

        axs[0].plot(np.ones(coords[vi]["nz"]) * prx / 1000, z / 1000, "--k")

        cf0 = axs[0].contourf(
            x / 1000,
            z / 1000,
            ds,
            np.linspace(-0.16, 0, 9),
            cmap="viridis_r",
            extend="both",
        )

        axs[0].contour(
            x / 1000,
            z / 1000,
            psi,
            np.linspace(-10, 10, 21),
            colors="white",
            alpha=0.2,
            linewidth=0.5,
        )
        axs[0].plot(x / 1000, dep / 1000, "k", linewidth=2)
        axs[0].plot(x / 1000, ice / 1000, "k", linewidth=2)
        axs[0].set_title("Exp: {} - ave. Day {}-{}".format(path[vi], time[0], time[-1]))
        axs[0].set_xlabel("Distance from grounding line [km]")
        axs[0].set_ylabel("depth in [km]")

        fwf = SHIflx[vi]["fwfx"] / 1000
        fwf[ice > 0] = np.nan
        meltflux = np.nansum(fwf) / 1000 * dx
        melt = np.nanmean(fwf) / 1000 * 86400 * 365
        # axs[0].text(
        #     1,
        #     -0.1,
        #     r"Melt flux: ${:7.3f} m^2/s$".format(meltflux),
        #     fontsize=12,
        # )
        axs[0].text(
            1,
            -0.1,
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
            sgrid,
            np.nancumsum(psi_s, axis=0),
            np.linspace(-10, 10, 21),
            cmap="RdBu_r",
            extend="both",
        )
        axs[1].set_xlabel("Distance from grounding line [km]")
        axs[1].set_ylabel("Sigma")
        axs[1].set_xlim([0, 30])
        # axs[1].set_ylim([34.9, 35])
        axs[1].invert_yaxis()
        cbar2 = fig.colorbar(cf1, ax=axs[1], orientation="horizontal", fraction=0.08)

        cbar1.set_label("Salinity")
        cbar2.set_label("Transport Stream function")
        plt.tight_layout()
        fig.savefig("plots/" + path[vi] + figname, facecolor="white", dpi=600)
        plt.show()
        plt.close("all")

    if "sec1" in which:
        figname = "_sec1.png"

        (t, s, u, w) = loadMIT.ave_tsu(
            data[vi]["t_all"], data[vi]["s_all"], data[vi]["u_all"], data[vi]["w_all"]
        )
        ds = np.zeros(np.shape(s))
        for i in np.arange(0, np.shape(s)[1]):
            ds[:, i] = s[:, i] - sref

        psi = np.flipud(np.nancumsum(np.flip(u, axis=0) * dz, axis=0))
        # nan above the ice
        for i in np.arange(1, coords[vi]["nx"]):
            t[z > ice[i], i] = float("nan")
            s[z > ice[i], i] = float("nan")
            u[z > ice[i], i] = float("nan")
            w[z > ice[i], i] = float("nan")
            psi[z > ice[i], i] = float("nan")
        r = rnil * (1 + tAlpha * t + sBeta * s) - 1000

        rgrid = np.linspace(np.nanmin(r), np.nanmax(r), 101)
        sgrid = np.linspace(34.9, np.nanmax(s), 21)
        psi_r = np.zeros([np.shape(rgrid)[0], np.shape(x)[0]])
        psi_s = np.zeros([np.shape(sgrid)[0], np.shape(x)[0]])

        for i in np.arange(0, len(x)):
            for ri in np.arange(0, len(rgrid)):
                ind = np.nonzero(r[:, i] > rgrid[ri])[0]
                if np.any(ind):
                    psi_r[ri, i] = np.nansum(np.flip(u[ind, i], axis=0), axis=0) * -dz
            for si in np.arange(0, len(sgrid) - 1):
                sind = np.nonzero((s[:, i] >= sgrid[si]) & (s[:, i] < sgrid[si + 1]))[0]
                if np.any(sind):
                    psi_s[si, i] = np.nansum(np.flip(u[sind, i], axis=0), axis=0) * -dz

        print(np.nanmax(psi_s))

        [fig, axs] = plt.subplots(1, 1, figsize=(8, 4))

        axs.plot(np.ones(coords[vi]["nz"]) * prx / 1000, z / 1000, "--k")
        cmap = cm.viridis_r
        bounds = np.arange(34.9, 35, 0.005)
        print(bounds)
        bounds = np.concatenate((np.arange(34, 34.9, 0.1), bounds))
        print(bounds)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend="both")
        cf0 = axs.contourf(
            x / 1000,
            z / 1000,
            s,
            levels=bounds,
            extend="both",
            cmap=cmap,
            norm=norm,
        )
        axs.contour(
            x / 1000,
            z / 1000,
            s,
            [34.9],
            colors="white",
        )

        axs.contour(
            x / 1000,
            z / 1000,
            -psi,
            np.linspace(0, 10, 5),
            colors="black",
            alpha=0.2,
            linewidth=0.5,
        )
        axs.plot(x / 1000, dep / 1000, "k", linewidth=2)
        axs.plot(x / 1000, ice / 1000, "k", linewidth=2)
        axs.set_title("Exp: {} - ave. Day {}-{}".format(path[vi], time[0], time[-1]))
        axs.set_xlabel("Distance from grounding line [km]")
        axs.set_ylabel("depth in [km]")

        fwf = SHIflx[vi]["fwfx"]
        fwf[ice > 0] = np.nan
        meltflux = np.nansum(fwf) / 1000 * dx
        melt = np.nanmean(fwf) / 1000 * 86400 * 365
        # axs[0].text(
        #     1,
        #     -0.1,
        #     r"Melt flux: ${:7.3f} m^2/s$".format(meltflux),
        #     fontsize=12,
        # )
        axs.text(
            1,
            -0.1,
            r"Ave melt rate: ${:5.1f} m/yr$".format(melt),
            fontsize=12,
        )
        ax02 = axs.twinx()
        ax02.plot(x / 1000, fwf, color="green")
        ax02.set_ylabel("fresh water flux")
        ax02.tick_params(axis="y", colors="green")
        ax02.spines["right"].set_color("green")
        ax02.ticklabel_format(axis="y", style="sci", scilimits=[-1, 2])
        ax02.set_xlim(0, 30)
        ax02.set_ylim(2 * np.nanmin(fwf), 0)
        cbar1 = fig.colorbar(
            cf0,
            ax=ax02,
            orientation="horizontal",
            fraction=0.08,
            ticks=[34, 34.9, 35],
        )  # , anchor=(1.0, 0.1)

        cbar1.set_label("Salinity")
        plt.tight_layout()
        fig.savefig("plots/" + path[vi] + figname, facecolor="white", dpi=600)
        plt.show()
        plt.close("all")

    if "delta" in which:
        figname = "_sec_del.png"
        dels = data[vi]["s_all"][-1, :, :] - data[vi]["s_all"][0, :, :]
        delt = data[vi]["t_all"][-1, :, :] - data[vi]["t_all"][0, :, :]

        idels = (
            np.nansum(dels[:, 0:pri] * hfac[:, 0:pri] * dx * dz)
            / (time[-1] - time[0] + 1)
            / 86400
            / saw
        )

        fig1 = plt.figure(figsize=(8, 8))
        gs1 = GridSpec(2, 1, figure=fig1)
        ax11 = fig1.add_subplot(gs1[0, 0])
        ax21 = fig1.add_subplot(gs1[1, 0])

        cf1 = ax11.contourf(
            x / 1000,
            z / 1000,
            dels,
            np.linspace(-0.02, 0.02, 21),
            cmap="RdBu_r",
            extend="both",
        )
        ax11.plot(x / 1000, dep / 1000, "k", linewidth=2)
        ax11.plot(x / 1000, ice / 1000, "k", linewidth=2)

        cbar1 = fig1.colorbar(cf1, ax=ax11, orientation="horizontal", fraction=0.08)
        cbar1.set_label("Temp diff")
        ax11.set_title("Exp: {} - diff. Day {}-{}".format(path[vi], time[-1], time[0]))

        cf2 = ax21.contourf(
            x / 1000,
            z / 1000,
            delt,
            np.linspace(-0.3, 0.3, 21),
            cmap="RdBu_r",
            extend="both",
        )
        ax21.text(1, -0.1, "Integrated dS/dt = {}".format(idels))
        ax21.plot(x / 1000, dep / 1000, "k", linewidth=2)
        ax21.plot(x / 1000, ice / 1000, "k", linewidth=2)

        cbar2 = fig1.colorbar(cf2, ax=ax21, orientation="horizontal", fraction=0.08)
        cbar2.set_label("Sal diff")

        fig1.savefig("plots/" + path + figname, facecolor="white", dpi=600)

    if "salflx" in which:
        figname = "_sec_sflx.png"

        s = np.nanmean(data[vi]["s_all"], axis=0)
        t = np.nanmean(data[vi]["t_all"], axis=0)
        u = np.nanmean(data[vi]["u_all"], axis=0)
        dels = np.zeros(np.shape(s))
        delt = np.zeros(np.shape(s))

        for i in np.arange(0, len(x)):
            dels[:, i] = s[:, i] - sref
            delt[:, i] = t[:, i] - tref

        sflx = np.flipud(np.nancumsum(np.flipud(u * s / saw), axis=0)) * dz
        tflx = np.flipud(np.nancumsum(np.flipud(u * t / taw), axis=0)) * dz

        for i in np.arange(1, coords[vi]["nx"]):
            sflx[z > ice[i], i] = float("nan")
            tflx[z > ice[i], i] = float("nan")

        fig1 = plt.figure(figsize=(8, 8))
        gs1 = GridSpec(2, 1, figure=fig1)
        ax11 = fig1.add_subplot(gs1[0, 0])
        ax21 = fig1.add_subplot(gs1[1, 0])

        cf1 = ax11.contour(x / 1000, z / 1000, sflx, [-0.05], colors="k")
        # ax11.plot(x / 1000, dep / 1000, "k", linewidth=2)
        # ax11.plot(x / 1000, ice / 1000, "k", linewidth=2)

        ax11.set_title(
            "Exp: {} - Sal transport ave Day {}-{}".format(path[vi], time[0], time[-1])
        )
        ax11.set_xlim([0, 30])

        cf2 = ax21.contour(x / 1000, z / 1000, tflx, np.arange(-10, 10, 2), colors="k")
        ax21.plot(x / 1000, dep / 1000, "k", linewidth=2)
        ax21.plot(x / 1000, ice / 1000, "k", linewidth=2)

        ax21.set_title(
            "Exp: {} - Temp transport ave Day {}-{}".format(path[vi], time[0], time[-1])
        )
        ax21.set_xlim([0, 30])

        fig1.savefig("plots/" + path[vi] + figname, facecolor="white", dpi=600)


def plot_prof(fig_name, path, prx, Gade=False, SalT=True):

    data = TM.data
    coords = TM.coords
    dx = coords[0]["dx"][0]
    t, s, u, w = TM.init_uts(data)
    upr, spr, tpr = TM.init_prof(coords[0], data, prx)
    dels = np.zeros(np.shape(upr))

    Q = np.zeros([np.shape(data)[0], np.shape(prx)[0], coords[0]["nz"]])
    mot = np.zeros([np.shape(data)[0], np.shape(prx)[0]])
    maxi = np.zeros([np.shape(data)[0], np.shape(prx)[0]])
    x = coords[0]["x"]
    sref = data[0]["sref"]
    saw = sref[-1]
    dz = -coords[0]["dz"][0]
    dy = coords[0]["dx"][0]
    hfac = coords[0]["hfac"]
    hfacx = np.zeros(len(coords[0]["ice"]))
    have = np.nanmean(hfac, axis=0)
    for i in np.arange(0, len(have)):
        ind = np.where((hfac[:, i] > 0) & (hfac[:, i] < 1))[0]
        if len(ind) < 1:
            hfacx[i] = np.max(hfac[:, i])
        else:
            hfacx[i] = hfac[ind, i]

    sflx = np.zeros(np.shape(data)[0])
    melt = np.zeros(np.shape(data)[0])
    taw = np.zeros(np.shape(data)[0])
    idels = np.zeros(np.shape(data)[0])

    for vi in np.arange(0, np.shape(data)[0]):
        time = data[vi]["time"]
        if time[-1] / 86400 > 100:
            time = time / 2

        t[vi, :, :], s[vi, :, :], u[vi, :, :], w[vi, :, :] = loadMIT.ave_tsu(
            data[vi]["t_all"], data[vi]["s_all"], data[vi]["u_all"], data[vi]["w_all"]
        )
        t = np.array(t)
        s = np.array(s)
        u = np.array(u)

        dels = data[vi]["s_all"][-1, :, :] - data[vi]["s_all"][0, :, :]

        melt[vi] = np.nansum(np.abs(TM.SHIflx[vi]["fwfx"] * hfacx)) / 1000 * dx

        for i in np.arange(0, np.shape(prx)[0]):
            pri = np.min(np.where(coords[vi]["x"] >= prx[i])[0])
            upr[vi, i, :] = u[vi, :, pri]
            tpr[vi, i, :] = t[vi, :, pri]
            spr[vi, i, :] = s[vi, :, pri]
            cond = np.isnan(upr[vi, i, :]).all()
            maxi[vi, i] = np.nanargmax(upr[vi, i, :]) if not cond else np.nan
            mot[vi, i], Q[vi, i, :] = diagnostics.ot_time_Q(
                coords[vi], upr[vi, i], Qout=True
            )

        idels[vi] = (
            np.nansum(dels[:, 0:pri] * hfac[:, 0:pri] * dx * dz)
            / (time[-1] - time[0])
            / saw
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
    sflx = np.squeeze(np.nansum(spr * upr, axis=2) / saw * dz)
    tflx = np.squeeze(np.nansum(tpr * upr, axis=2) / taw * dz)

    tAlpha = (-0.4e-4,)
    sBeta = (8.0e-4,)
    mot = np.array(mot)
    maxi = np.array(maxi)
    Q = np.array(Q)
    upr = np.array(upr)
    tpr = np.array(tpr)
    spr = np.array(spr)

    fig = plt.figure(figsize=(8, 6))
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    # ax5 = plt.subplot(3, 2, 5)
    # ax6 = plt.subplot(3, 2, 6)
    axs = [ax1, ax2, ax3, ax4]

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

    # axs[4].grid("both")
    ## axs[4].set_ylim(-1.050, 0.050)
    # axs[4].set_ylabel("ransport [m^2/s]")
    # axs[4].set_xlabel("$T_{AW}$")
    #
    # axs[5].grid("both")
    # axs[5].set_ylabel("Volume transport [$m^2/s$]")
    # axs[5].set_xlabel("Salinity")
    # axs[5].set_xlim(34.8, 35.0)

    for vi in np.arange(0, np.shape(data)[0]):
        i = 0
        rpr = 999.8 * (1 + tAlpha * tpr[vi, i, :] + sBeta * spr[vi, i, :]) - 1000
        rmax = rpr[np.nanargmax(upr[vi, i, :])]
        z = TM.coords[vi]["z"]
        zmax = TM.coords[vi]["z"][np.nanargmax(upr[vi, i, :])]
        tref = TM.data[vi]["tref"]
        taw[vi] = tref[-1]

        tsdata = [z, np.squeeze(tpr[vi, i, :]), np.squeeze(spr[vi, i, :])]
        tsref = pd.DataFrame(
            columns=["z", "tref", "sref"], data=np.flipud(np.transpose(tsdata))
        )
        tsref.to_csv("tsref_{}".format(path[vi]), index=False)
        tsref.to_csv(
            "../PlumeModel/tsref_{}".format(path[vi]), index=False, header=False
        )

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
            # axs[4].plot(
            #    tref[-1],
            #    -sflx[vi, i],
            #    marker,
            #    color=color,
            # )
            # axs[5].plot(spr[vi, 0, :], upr[vi, 0, :], color=color)

    # axs[4].plot(taw, melt, label="melt")
    # axs[4].plot(taw, -sflx, label="hor flux at XXkm")
    # axs[4].plot(taw, idels, label="d/dt fresh water content")
    # axs[4].legend()

    plt.tight_layout()
    axs[1].legend(loc="lower center", bbox_to_anchor=(0.35, 0.0), ncol=2)
    fig.savefig("plots/" + fig_name, facecolor="white", dpi=600)
    fig.show()


def plot_plume(figname, path, which=[], prx=21e3):
    importlib.reload(diagnostics)

    global mdata
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
    x = coords[0]["x"]
    dx = coords[0]["dx"][0]
    ice = coords[0]["ice"]

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
        ax21.set_xlim(0, 25)
        ax21.set_ylim(0, 75)

        ax22.set_ylabel("melt from ice")
        ax22.set_xlabel("dist along ice")
        # ax22.set_xticklabels([])
        ax22.grid("both")
        ax22.set_xlim(0, 25)

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
        fig3 = plt.figure(figsize=(6, 4))
        gs3 = GridSpec(1, 2, figure=fig3)
        ax31 = fig3.add_subplot(gs3[0, 0])
        ax32 = fig3.add_subplot(gs3[0, 1])

        ax31.set_xlabel("Plume Buyancy")
        ax31.set_ylabel("depth")
        ax31.grid("both")
        ax31.set_ylim(-1001, 1)

        ax32.set_xlabel(r"$\Delta \rho$")
        ax32.set_yticklabels([])
        ax32.grid("both")
        # ax32.set_xlim(34.79, 35.01)
        ax32.set_ylim(-1001, 1)

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
        ax41.ticklabel_format(axis="y", style="sci", scilimits=[-1, 2])
        #
        ax42.set_title("ave Melt [m^2/s]")
        # ax42.set_ylim(0, 0.075)
        ax42.set_xlim(-2.51, 6.01)
        ax42.grid("both")
        ax42.set_xticklabels([])
        ax42.ticklabel_format(axis="y", style="sci", scilimits=[-1, 2])
        #
        ax43.set_title("sum(Melt) / QP")
        ax43.set_ylim(0, 5e-3)
        ax43.set_xlim(-2.51, 6.01)
        ax43.grid("both")
        ax43.set_xticklabels([])
        ax43.ticklabel_format(axis="y", style="sci", scilimits=[-1, 2])

        ax44.set_title("$b_{total}$")
        ax44.set_ylabel("Plume Buoyancy")
        ax44.set_xlabel("AW Temperature")
        # ax44.set_ylim(-0.01, 0.61)
        ax44.set_xlim(-2.51, 6.01)
        ax44.grid("both")

        ax45.set_title("$b_{Sal}$")
        ax45.set_xlabel("AW Temperature")
        # ax45.set_ylim(-0.01, 0.61)
        ax45.set_xlim(-2.51, 6.01)
        ax45.grid("both")

        ax46.set_title("$b_{Temp}$")
        ax46.set_xlabel("AW Temperature")
        # ax46.set_ylim(-0.61, 0.01)
        ax46.set_xlim(-2.51, 6.01)
        ax46.grid("both")

    if "select1" in which:
        fig5 = plt.figure(figsize=(8, 4))
        gs5 = GridSpec(2, 1, figure=fig5)
        ax51 = fig5.add_subplot(gs5[0, 0])
        ax52 = fig5.add_subplot(gs5[1, 0])
        # ax53 = fig5.add_subplot(gs5[2, 0])
        ax51.set_xticklabels([])
        ax51.set_ylabel("Plume \nthickness [m]")
        ax51.grid("both")
        ax51.set_xlim(0, 15)
        ax51.set_xticks(range(15))
        ax51.set_ylim(-2, 125)

        # ax52.set_xticklabels([])
        ax52.set_xlabel("Distance along Ice [km]")
        ax52.set_ylabel("BL veloctiy [m/s]")
        ax52.grid("both")
        ax52.set_xlim(0, 15)
        ax52.set_xticks(range(15))
        # ax52.set_ylim(-0.01, 0.17)

        # ax52.set_xticklabels([])
        # ax53.set_xlabel("Distance along Ice [km]")
        # ax53.set_ylabel("melt rate [m/s]")
        # ax53.grid("both")
        # ax53.set_xlim(0, 15)
        # ax53.set_xticks(range(15))
        # ax53.set_ylim(None, 5e-6)

    if "select2" in which:
        fig6 = plt.figure(figsize=(8, 4))
        gs6 = GridSpec(1, 2, figure=fig6)
        ax61 = fig6.add_subplot(gs6[0, 0])
        ax62 = fig6.add_subplot(gs6[0, 1])

        ax61.set_title("Ave Melt $[m/yr]$")
        ax61.grid("both")
        ax61.set_xlabel("AW Temperature")

        ax62.set_title("$U_{BL} [m/s]$")
        ax62.grid("both")
        ax62.set_xlabel("AW Temperature")

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

    if "TSonly" in which:
        fig9 = plt.figure(figsize=(6, 6))
        gs9 = GridSpec(1, 1, figure=fig9)
        ax91 = fig9.add_subplot(gs9[0, 0])
        ax91.set_ylim(-0.1, 0.2)
        ax91.set_xlim(34.85, 35.001)

        ax91.grid("both")
        ax91.set_ylabel("Temp")
        ax91.set_xlabel("Sal")
        t, s, rm = diagnostics.rho_cont([-0.1, 0.2], [34.85, 35.001])
        ax91.contour(s, t, rm, 8, colors="k", alpha=0.2, linewidth=1)

    if "thick" in which:
        heights = [3, 3, 3, 3]
        fig10 = plt.figure(figsize=(8, 8))
        gs1 = GridSpec(4, 1, figure=fig10, height_ratios=heights)
        ax101 = fig10.add_subplot(gs1[0, :])
        ax102 = fig10.add_subplot(gs1[1, :])
        ax103 = fig10.add_subplot(gs1[2, :])
        ax104 = fig10.add_subplot(gs1[3, :])

        ax101.set_ylabel("thickness\nw > 0")
        ax101.grid("both")
        ax101.set_xlim(0, 15)
        ax101.set_ylim(0, 120)
        ax101.set_xticklabels([])

        ax102.set_xticklabels([])
        ax102.set_ylabel("thickness\nt - t_amb < -0.005")
        ax102.grid("both")
        ax102.set_xlim(0, 15)
        ax102.set_ylim(0, 120)

        ax103.set_xticklabels([])
        ax103.set_ylabel("thickness\ns - s_amb < -0.001")
        ax103.grid("both")
        ax103.set_ylim(0, 120)
        ax103.set_xlim(0, 15)

        ax104.set_ylabel("thickness\n u > 0")
        ax104.set_xlabel("distance along Ice [km]")
        ax104.grid("both")
        ax104.set_xlim(0, 15)
        ax104.set_ylim(0, 120)

    for vi in np.arange(0, np.shape(data)[0]):

        plumew = diagnostics.plume(coords[vi], data[vi], ["flx"], mask="w")
        plumeu = diagnostics.plume(coords[vi], data[vi], ["flx"], mask="u")
        plumet = diagnostics.plume(coords[vi], data[vi], ["flx"], mask="t")
        plumes = diagnostics.plume(coords[vi], data[vi], ["flx"], mask="s")
        t_plu = plumet["t_plu"]
        s_plu = plumes["s_plu"]
        u_plu = plumeu["u_plu"]
        w_plu = plumeu["w_plu"]
        u_plu_all = plumeu["u_plu_all"]
        w_plu_all = plumeu["w_plu_all"]
        z_plu = plumeu["z_plu"]
        pri = np.min(np.where(coords[vi]["x"] >= prx)[0])
        tpr = TM.data[vi]["t_all"][vi, :, pri]
        spr = TM.data[vi]["s_all"][vi, :, pri]

        x = np.squeeze(plumew["d"] / 1000)
        dw = plumew["thick"]
        du = plumeu["thick"]
        dt = plumet["thick"]
        ds = plumes["thick"]
        taw = data[vi]["tref"][-1]
        saw = data[vi]["sref"][-1]
        ice = coords[vi]["ice"]
        if len(ice) > len(x):
            ice = ice[: len(x)]
        tref = data[vi]["tref"]
        sref = data[vi]["sref"]
        t0 = 1
        s0 = 35

        melt = np.abs(SHIflx[vi]["fwfx"]) / 1000

        cmelt = np.nancumsum(melt * dx)
        amelt = np.nanmean(melt * dx)

        color, line, marker = TM.identify(path[vi], tref[-1])

        u_ice = plumew["u_plu"][0, :]
        w_ice = plumew["w_plu"][1, :]
        vel_ice = np.sqrt(u_ice**2 + w_ice**2)

        t_ice = plumew["t_plu"][0, :]
        t_ave = np.nanmean(t_plu, axis=0)
        s_ice = plumew["s_plu"][0, :]
        s_ave = np.nanmean(s_plu, axis=0)
        vel_plu = np.sqrt(u_plu**2 + w_plu**2)
        vel_plu_all = np.sqrt(u_plu_all**2 + w_plu_all**2)
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
        nans, i = diagnostics.nan_helper(dw)
        dw[nans] = np.interp(i(nans), i(~nans), dw[~nans])
        nans, i = diagnostics.nan_helper(du)
        du[nans] = np.interp(i(nans), i(~nans), du[~nans])
        nans, i = diagnostics.nan_helper(dt)
        dt[nans] = np.interp(i(nans), i(~nans), dt[~nans])
        nans, i = diagnostics.nan_helper(ds)
        ds[nans] = np.interp(i(nans), i(~nans), ds[~nans])

        sa_fil = uniform_filter1d(s_ave, size=20)
        si_fil = uniform_filter1d(s_ice, size=20)
        vi_fil = uniform_filter1d(vel_ice, size=20)
        va_fil = uniform_filter1d(vel_ave, size=20)
        ti_fil = uniform_filter1d(t_ice, size=20)
        ta_fil = uniform_filter1d(t_ave, size=20)
        dw_fil = uniform_filter1d(dw, size=20)
        du_fil = uniform_filter1d(du, size=20)
        dt_fil = uniform_filter1d(dt, size=20)
        ds_fil = uniform_filter1d(ds, size=20)

        t_f = lam1 * si_fil + lam2 + ice * lam3
        TF = t_ice - t_f

        rho1 = (1 + (sref) * sBeta + (tref) * tAlpha) * rnil
        rho2 = (1 + (sa_fil) * sBeta + (ta_fil) * tAlpha) * rnil

        raw = (1 + (saw) * sBeta + (taw) * tAlpha) * rnil

        rpr = ((spr) * sBeta + (tpr) * tAlpha + 1) * rnil
        ramb = np.interp(ice, z, rpr)
        tamb = np.interp(ice, z, tpr)
        samb = np.interp(ice, z, spr)

        rho2s = (1 + (tamb) * tAlpha + (sa_fil) * sBeta) * rnil
        rho2t = (1 + (ta_fil) * tAlpha + (samb) * sBeta) * rnil

        s_buo = (rho2s - raw) * g
        t_buo = (rho2t - raw) * g
        buo = (rho2 - raw) * g

        g_red = (rho1[-1] - rho2) / rho1[-1] * -g

        flx = plumew["flx"]
        flxi = np.min([np.nanargmax(flx), 1500])

        fr = va_fil / np.sqrt(g_red * dw_fil)
        ri = g_red * dw_fil / va_fil**2

        if "tsu" in which:
            ax11.plot(x, dw_fil, color=color, linestyle="-")
            # ax12.plot(x, ds_fil, color=color, linestyle="--")
            # ax13.plot(x, dt_fil, color=color, linestyle="-.")
            # ax14.plot(x, du_fil, color=color, linestyle=":")

            ax12.plot(
                x,
                si_fil - saw,
                color=color,
                linestyle=line,
                alpha=0.7,
            )
            ax13.plot(
                x,
                ti_fil - taw,
                color=color,
                linestyle=line,
                alpha=0.7,
                label=path[vi],
            )
            ax14.plot(x, vi_fil, color=color, linestyle=line, alpha=0.7, label=path[vi])
            ax14.plot(
                x, vel_ice, color=color, linestyle=line, alpha=0.1, label="_nolegend_"
            )

        # ------------Figure 2-------------
        if "flux" in which:
            ax21.plot(x, flx, color=color, linestyle=line, alpha=0.7, label=path[vi])
            ax22.plot(
                x,
                np.nancumsum(melt * dx),
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
            ax31.plot(
                buo[0 : np.nanargmax(flx)],
                ice[0 : np.nanargmax(flx)],
                linestyle="-",
                color=color,
            )
            ax32.plot(
                rpr,
                z,
                linestyle="--",
                color=color,
            )
            # ax32.plot(
            #     sref,
            #     z,
            #     linestyle=":",
            #     color=color,
            # )
            ax32.plot(
                rho2[0 : np.nanargmax(flx)],
                ice[0 : np.nanargmax(flx)],
                linestyle="-",
                color=color,
            )

        # ------------Figure 4-------------
        if "sum" in which:
            ax41.plot(taw, np.nanmax(flx), marker, color=color)
            ax42.plot(taw, cmelt[flxi], marker, color=color)
            ax43.plot(
                taw,
                cmelt[flxi] / np.nanmax(flx),
                marker,
                color=color,
            )
            ax44.plot(
                taw,
                np.nanmax(buo[0:flxi]),
                marker,
                color=color,
            )
            ax45.plot(
                taw,
                np.nanmax(s_buo[0:flxi]),
                marker,
                color=color,
            )
            ax46.plot(
                taw,
                np.nanmax(t_buo[0:flxi]),
                marker,
                color=color,
            )

        # ------------Figure 5-------------
        if "select1" in which:
            ax51.plot(x, du_fil, color=color, linestyle=line, label=path[vi])
            ax52.plot(x, vi_fil, color=color, linestyle=line, label=path[vi])
            # ax53.plot(x, melt, color=color, linestyle=line, label=path[vi])

        # ------------Figure 6-------------
        if "select2" in which:
            # ax61.plot(taw, cmelt[np.nanargmax(flx)], marker, color=color)
            ax61.plot(taw, amelt, marker, color=color)
            ax62.plot(
                taw, np.nanmean(vel_ice[0 : np.nanargmax(flx)]), marker, color=color
            )
            ax63.plot(taw, np.nanmean(TF[0 : np.nanargmax(flx)]), marker, color=color)

        # ------------Figure 7-------------
        if "nondim" in which:
            ax71.plot(x, fr, color=color, linestyle=line, label=path[vi])
            ax72.plot(x, ri, color=color, linestyle=line, label=path[vi])

        if "UWTSsec" in which:
            # fig8 = plt.figure(figsize=(8, 6))
            # gs8 = GridSpec(4, 1, figure=fig8)
            fig8 = plt.figure(figsize=(8, 10))
            gs8 = GridSpec(6, 1, figure=fig8)
            ax81 = fig8.add_subplot(gs8[0, 0])
            ax82 = fig8.add_subplot(gs8[1, 0])
            ax83 = fig8.add_subplot(gs8[2, 0])
            ax84 = fig8.add_subplot(gs8[3, 0])
            ax85 = fig8.add_subplot(gs8[4, 0])
            ax86 = fig8.add_subplot(gs8[5, 0])

            ax81.text(1, -100, "Time Averaged U Velocity", fontweight="bold")
            ax81.set_xlim(0, 15)
            ax81.set_xticklabels([])
            ax81.set_ylabel("orth. dist.\nfrom ice [m]")

            ax82.set_xlim(0, 15)
            ax82.set_xticklabels([])
            ax82.text(1, -100, "Standard Deviation of U Velocity", fontweight="bold")
            ax82.set_ylabel("orth. dist.\nfrom ice [m]")

            ax83.text(1, -100, "Time Averaged W Velocity", fontweight="bold")
            ax83.set_xlim(0, 15)
            ax83.set_xticklabels([])
            ax83.set_ylabel("orth. dist.\nfrom ice [m]")

            ax84.set_xlim(0, 15)
            ax84.set_xticklabels([])
            ax84.text(1, -100, "Standard Deviation of W Velocity", fontweight="bold")
            ax84.set_ylabel("orth. dist.\nfrom ice [m]")

            ax85.set_xlim(0, 15)
            ax85.set_xticklabels([])
            ax85.text(1, -100, "Plume Temperature", fontweight="bold")
            ax85.set_ylabel("orth. dist.\nfrom ice [m]")

            ax86.set_xlim(0, 15)
            ax86.set_xlabel("Distance along Ice [km]")
            ax86.text(1, -100, "Plume salinity", fontweight="bold")
            ax86.set_ylabel("orth. dist.\nfrom ice [m]")

            cf1 = ax81.contourf(
                x,
                z_plu,
                u_plu,
                levels=np.linspace(0, 0.2, 11),
                extend="max",
            )
            cf2 = ax82.contourf(
                x,
                z_plu,
                np.std(u_plu_all, axis=0),
                levels=np.linspace(0, 1e-2, 21),
                extend="max",
            )
            cf3 = ax83.contourf(
                x,
                z_plu,
                w_plu,
                levels=np.linspace(0, 0.02, 11),
                extend="both",
            )
            cf4 = ax84.contourf(
                x,
                z_plu,
                np.std(w_plu_all, axis=0),
                levels=np.linspace(0, 1e-3, 21),
            )
            cf5 = ax85.contourf(
                x,
                z_plu,
                plumeu["t_plu"] - taw,
                levels=np.linspace(-0.2, 0, 11),
                extend="both",
            )
            cf6 = ax86.contourf(
                x,
                z_plu,
                plumeu["s_plu"] - saw,
                levels=np.linspace(-0.1, 0, 11),
                extend="both",
            )

            fig8.colorbar(cf1, ax=ax81)
            fig8.colorbar(cf2, ax=ax82)
            fig8.colorbar(cf3, ax=ax83)
            fig8.colorbar(cf4, ax=ax84)
            fig8.colorbar(cf5, ax=ax85)
            fig8.colorbar(cf6, ax=ax86)

            fig8.tight_layout()
            fig8.savefig(
                "plots/plume_TSUWsec_{}.png".format(path[vi]),
                facecolor="white",
                dpi=600,
            )
            # fig8.savefig(
            #    "plots/plume_UWsec_{}.png".format(path[vi]),
            #    facecolor="white",
            #    dpi=600,
            # )

        if "TSonly" in which:
            SG, TG = diagnostics.gadeline(tam=taw, z=1000)
            # ax91.plot(
            #    SG,
            #    TG - taw,
            #    "-",
            #    linewidth=1,
            # )

            # ax91.scatter(
            #     sa_fil[1 : flxi],
            #     ta_fil[1 : flxi],
            #     10,
            #     marker=".",
            #     linewidth=0,
            #     #alpha=0.3,
            #     label=path[vi]
            # )
            ax91.plot(spr, tpr, "--", linewidth=1, label=path[vi])

        if "thick" in which:
            ax101.plot(x, dw_fil, color=color, linestyle="--")
            ax102.plot(x, dt_fil, color=color, linestyle="--")
            ax103.plot(x, ds_fil, color=color, linestyle="--")
            ax104.plot(x, du_fil, color=color, linestyle="--")

    mdata = diagnostics.mdata
    mdata.to_csv("meltdata")

    if "tsu" in which:
        fig1.tight_layout()
        ax14.legend(loc="center right", ncol=1, bbox_to_anchor=(1.00, 1.5))
        fig1.savefig("plots/tsu" + figname, facecolor="white")

    if "flux" in which:
        fig2.tight_layout()
        ax22.legend(loc="center right", ncol=4)  # , bbox_to_anchor=(1.00, 1))
        fig2.savefig("plots/flux" + figname, facecolor="white")

    if "buoy" in which:
        fig3.tight_layout()
        ax31.legend(loc="lower center", ncol=3, bbox_to_anchor=(-0, 0))
        fig3.savefig("plots/buoy" + figname, facecolor="white", dpi=600)

    if "sum" in which:
        fig4.tight_layout()
        # ax46.legend(loc="lower center", ncol=3, bbox_to_anchor=(-0, 0))
        fig4.savefig("plots/sum" + figname, facecolor="white")

    if "select1" in which:
        fig5.tight_layout()
        # ax51.legend(loc="upper left", ncol=3)  # , bbox_to_anchor=(0.5, 0))
        fig5.savefig("plots/DTU" + figname, facecolor="white", dpi=600)

    if "select2" in which:
        fig6.tight_layout()
        # ax61.legend(loc="lower center", ncol=1, bbox_to_anchor=(0.5, 0))
        fig6.savefig("plots/Melt" + figname, facecolor="white", dpi=600)

    if "nondim" in which:
        fig7.tight_layout()
        ax71.legend(loc="lower center", ncol=4)
        fig7.savefig("plots/nondim" + figname, facecolor="white", dpi=600)

    if "TSonly" in which:
        fig9.tight_layout()
        ax91.legend(loc="lower center", ncol=2)
        fig9.savefig("plots/TSonly" + figname, facecolor="white", dpi=600)

    if "thick" in which:
        axx101 = ax101.twiny()
        tickloc = ax101.get_xticks()
        ticklab = np.zeros(len(tickloc))
        for i in range(len(tickloc)):
            ind = np.where(x > tickloc[i])[0]
            ticklab[i] = np.floor(ice[ind[0]])
        axx101.set_xticks(tickloc)
        axx101.set_xlim(0, 15)
        axx101.set_xticklabels(ticklab)
        axx101.set_xlabel("ice depth [m]")
        fig10.tight_layout()
        fig10.savefig("plots/D" + figname, facecolor="white", dpi=600)


def plot_ekin(figname, path, flx=[], prx=[]):

    pri = np.zeros(np.shape(prx)[0]).astype("int")

    [fig, axs] = plt.subplots(4, 1, figsize=(8, 6))

    print("start plotting {} timeseries".format(np.shape(path)[0]))
    for vi in np.arange(0, np.shape(path)[0]):
        print("plot {} of {}".format(vi + 1, np.shape(path)[0]))
        data, coords, SHIflx, gammas = TM.load_single(path[vi], "times")

        time = data["time"] / 86400
        step = np.squeeze(np.arange(0, np.shape(time)[0]) + 1)
        dz = -coords["dz"][0]
        dx = coords["dx"][0]
        saw = data["sref"][-1]

        color, line, marker = TM.identify(path[vi], data["tref"][-1])
        u = data["u_all"]
        w = data["w_all"]
        t = data["t_all"]
        s = data["s_all"]
        hfac = coords["hfac"]
        hfacx = np.zeros(len(coords["ice"]))
        have = np.nanmean(hfac, axis=0)
        for i in np.arange(0, len(have)):
            ind = np.where((hfac[:, i] > 0) & (hfac[:, i] < 1))[0]
            if len(ind) < 1:
                hfacx[i] = np.max(hfac[:, i])
            else:
                hfacx[i] = hfac[ind, i]

        s_int = np.nansum(s[:, :, :] * hfac, axis=(1, 2)) / np.sum(hfac, axis=(0, 1))
        t_int = np.nansum(t[:, :, :] * hfac, axis=(1, 2)) / np.sum(hfac, axis=(0, 1))
        ekin = np.nansum(u * u + w * w, axis=(1, 2))
        motQ = np.zeros(np.shape(time))
        sflx = np.zeros(np.shape(time))
        melt = np.zeros(np.shape(time))

        print(np.shape(SHIflx["fwf_all"]))

        pri = int(str(min(np.where(coords["x"] >= prx)[0])))
        for n in np.arange(0, np.shape(data["time"])[0]):
            upr = data["u_all"][n, :, pri]
            spr = data["s_all"][n, :, pri]
            if SHIflx is None:
                melt[n] = 0
            else:
                melt[n] = np.nansum(SHIflx["fwf_all"][n] * hfacx) / 1000 * dx

            motQ[n] = diagnostics.ot_time_Q(coords, upr)
            sflx[n] = np.squeeze(np.nansum(spr * upr) / saw * dz)

        motQ = np.array(motQ)

        if "AW02" in path[vi]:
            label = "ryder"
            line = "--"
            color = "k"
        else:
            label = path[vi][12:]

        axs[0].plot(
            step,
            ekin,
            linestyle=line,
            color=color,
            label=label,
        )

        axs[2].plot(
            step,
            melt,
            line,
            color=color,
        )

        axs[2].plot(
            step,
            sflx,
            ":",
            color=color,
        )

        axs[1].plot(
            step,
            motQ,
            line,
            color=color,
        )

        axs[3].plot(
            step,
            (t_int - t_int[40]),
            ":",
            color=color,
        )

        axs[3].plot(
            step,
            (s_int - s_int[40]),
            "-",
            color=color,
        )

    axs[0].set_xticklabels([])
    axs[0].set_ylabel("Kinetic Energy")
    axs[0].grid("both")
    axs[0].legend(ncol=3, loc=8, fontsize="small")
    axs[0].set_xlim([-1, 101])

    axs[2].set_xticklabels([])
    axs[2].set_ylabel("melt vs. FWF\n[m^2/s]")
    axs[2].grid("both")
    axs[2].set_xlim([-1, 101])
    axs[2].ticklabel_format(axis="y", style="sci", scilimits=[-1, 2])

    axs[1].set_xticklabels([])
    axs[1].set_ylabel("Fjord Overturngin\n timescale [d]")
    axs[1].grid("both")
    axs[1].set_ylim([18, 32])
    axs[1].set_xlim([-1, 101])

    axs[3].set_xlabel("Model Days")
    axs[3].set_ylabel("Integrated\n T and S")
    axs[3].grid("both")
    axs[3].set_xlim([-1, 101])
    axs[3].set_ylim([-0.005, 0.015])
    axs[3].ticklabel_format(axis="y", style="sci", scilimits=[-1, 2])

    plt.tight_layout()
    plt.show()
    fig.savefig("plots/all" + figname, facecolor="white", dpi=600)
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


def plot_melt(figname):
    mdata = diagnostics.mdata
    rmse = np.zeros(len(mdata) - 2)
    rsq = np.zeros(len(mdata) - 2)

    for i in range(len(mdata) - 2):
        drop = list(range(i))
        data = mdata.drop(index=drop)
        model = smf.ols("MeltFlux ~ T_AW", data=data)
        fit = model.fit()
        rmse[i] = np.sqrt(np.sum(fit.resid**2))
        rsq[i] = fit.rsquared

    diff = np.diff(rmse)
    drop = np.nanargmax(rsq)
    data = mdata.drop(index=list(range(0, drop)))
    model = smf.ols("MeltFlux ~ T_AW", data=data)
    fit = model.fit()
    mmodel = smf.ols("MeltFlux ~ T_AW", data=mdata)
    mfit = mmodel.fit()

    fig = plt.figure(figsize=(12, 3))
    gs1 = GridSpec(1, 3, figure=fig)
    ax1 = fig.add_subplot(gs1[:, 0])
    ax2 = fig.add_subplot(gs1[:, 1])
    ax3 = fig.add_subplot(gs1[:, 2])

    ax1.plot(mdata["T_AW"], mdata["MeltFlux"], "kx")
    pl = ax1.plot(data["T_AW"], fit.predict(), linewidth=2)
    plm = ax1.plot(mdata["T_AW"], mfit.predict(), linewidth=2)
    ax1.set_xlabel("T_AW")
    ax1.set_ylabel("total Melt")

    ax22 = ax2.twinx()
    ax22.plot(mdata["T_AW"][0:-2], rsq, "ko", fillstyle="none")
    ax2.plot(mdata["T_AW"][0:-2], rmse, "kx")
    ax2.plot(mdata["T_AW"][drop] * np.ones(len(mdata) - 2), rmse, "k--")
    ax2.set_ylabel("RMSE (blue x)")
    ax22.set_ylabel("R**2 (black o)")
    ax2.set_xlabel("T_AW")
    ax2.ticklabel_format(axis="y", style="sci", scilimits=[-1, 2])
    ax22.ticklabel_format(axis="y", style="sci", scilimits=[-1, 2])
    ax2.set_xlim(ax1.get_xlim())

    ax3.plot(mdata["T_AW"], mdata["T_AW"] * 0, "--k")
    ax3.plot(mdata["T_AW"], mfit.resid, "x", label="all", color=plm[0].get_color())
    ax3.plot(data["T_AW"], fit.resid, "x", label="T_AW>=0", color=pl[0].get_color())
    ax3.set_ylabel("residuals")
    ax3.set_xlabel("T_AW")
    ax3.ticklabel_format(axis="y", style="sci", scilimits=[-1, 2])
    pos3 = ax3.get_position()
    ax3.set_position([pos3.x0 + 0.07, pos3.y0, pos3.x1 - pos3.x0, pos3.y1 - pos3.y0])
    ax3.legend()

    fig.tight_layout
    fig.savefig("plots/{}".format(figname), facecolor="white", dpi=600)
