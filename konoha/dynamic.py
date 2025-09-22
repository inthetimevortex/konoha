#!/usr/bin/env python
# -*- coding: utf-8 -*-
"Dynamic Spectra maker!!"
#  dynamic.py
#
#  Copyright 2018 Amanda Rubio <amanda.rubio@usp.br>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#

import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import pyhdust.spectools as spt
import pyhdust as phd
from glob import glob
from astropy.time import Time
from scipy.optimize import curve_fit
from matplotlib.colors import ListedColormap
import jdcal
from collections import OrderedDict
import seaborn as sns
import datetime as dt
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
from matplotlib.colorbar import Colorbar
import seaborn as sns
import copy
from icecream import ic
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.widgets import Slider, Button

sns.set_style("ticks", {"xtick.major.direction": "in", "ytick.major.direction": "in"})


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# vel_all, flux_all, MJD_all, flag_all = get_halpha()
def dynamic_spectra(
    line,
    MJD_all,
    vel_all,
    flux_all,
    resolution,
    vmin,
    vmax,
    set_limits,
    phase_folded,
    log_scale,
    P,
    t0,
    velmin,
    velmax,
    time_cut,
    axes,
    trange,
    subplot,
    save_plot,
):

    # resolution = 0.5 #days

    flx_all = []
    temp = np.arange(velmin, velmax, 1)

    MJD_to_sort = np.array(MJD_all)
    sort = MJD_to_sort.argsort()
    MJD_all = np.array(MJD_to_sort)[sort]
    # flag_all = np.array(flag_all)[sort]
    vel_all = np.array(vel_all)[sort]
    flux_all = np.array(flux_all)[sort]
    # print(len(MJD_all))
    # print(vel_all)
    # print(flux_all)
    for i in range(len(flux_all)):
        # interpolates specrum on 'temp' so they all have the same size :)
        flx = griddata(vel_all[i], flux_all[i], temp, method="linear")
        # plt.plot(vel_all[i], flux_all[i], linewidth=0.4, label=MJD_all[i])
        flx_all.append(flx)
    # print(flx_all)
    flxx = np.array(flx_all)
    keep = np.logical_not(np.isnan(flx_all))[:, 0]
    flx_all = flxx[keep]
    vel_all = vel_all[keep]
    MJD_all = MJD_all[keep]
    MJD_to_sort = np.array(MJD_all)
    sort = MJD_to_sort.argsort()
    MJD_a = MJD_all[sort]

    # plt.legend()
    # plt.show()
    # print(len(MJD_all))
    # im so good at names
    # hello = np.mean(flx_all, axis=0)
    # print(MJD_all[MJD_all < 2200])

    # hello = np.mean(flx_all[MJD_all < 2200], axis=0)
    hello = np.mean(flx_all[MJD_all > time_cut], axis=0)

    flx_all = flx_all[MJD_all > time_cut]
    MJD_a = MJD_a[MJD_all > time_cut]

    supes = []

    for j in range(len(flx_all)):
        # superflux = np.tile(flx_all[j]-hello, (1, 1))
        # keep = np.logical_and(vel_all[j] > -100, vel_all[j] < 100)
        mask_vel = np.logical_and(temp > -50, temp < 50)
        superflux = flx_all[j] - hello
        # mean_flux = np.mean(superflux[mask_vel])
        # superflux = superflux / np.abs(mean_flux)
        # superflux = np.tile(flx_all[j], (1, 1))
        # supes.append((superflux+ 1)**3 - 1)
        supes.append(superflux)

    # flux_a = np.tile(flx_all[0], (1, 1))
    # print(MJD_a)
    MJD_keep = np.copy(MJD_a)
    MJD_a = MJD_a - time_cut  # min(MJD_a)

    phase = (MJD_a - t0) / P % 1
    phase_keep = np.copy(phase)
    # ic(phase_keep)

    if not phase_folded:
        MJD_a = MJD_a / resolution

        # print(MJD_a)

        size_time = np.arange(0, (trange[1] - trange[0]) / resolution, 1)
        print(len(size_time))
        print(size_time)

        master = np.zeros([len(size_time), len(hello)])
        data_positions = []

        for k in range(len(MJD_a)):
            data_pos = int(MJD_a[k])
            data_positions.append(data_pos)

        pos_breakdown = list(set(data_positions))

        for final_pos in pos_breakdown:
            pos = np.where(np.array(data_positions) == final_pos)[0]
            flx_pos = np.sum(np.array(supes)[pos], axis=0) / len(pos)

            master[final_pos] = flx_pos

        # siz = max(MJD_a) / 150
        siz = (trange[1] - trange[0]) / resolution / 150
        s_master = np.zeros([150, len(hello)])
        # MJD_norm = MJD_a/max(MJD_a)
        # ic(MJD_a)
        # MJD_range = np.arange(MJD_keep[0], MJD_keep[-1], 96)
        for cc in range(0, 150):
            MJD_range = [int(siz) * cc, int(siz) * cc + 5.0]
            # ic(MJD_range)
            s_mask = np.ma.masked_inside(MJD_a, MJD_range[0], MJD_range[1]).mask
            # ic(MJD_keep[s_mask])
            try:
                s_flx_pos = np.sum(np.array(supes)[s_mask], axis=0) / len(
                    np.array(supes)[s_mask]
                )
                # ic(len(s_flx_pos))
            except RuntimeWarning:
                # if len(s_flx_pos) != len(hello):
                s_flx_pos = np.zeros(len(hello))
            s_master[cc] = s_flx_pos
        master = np.copy(master)
        # set_time_limits = [MJD_keep.max(), MJD_keep.min()]
        set_time_limits = [trange[1], trange[0]]
        # print(set_time_limits)
        set_time_label = "RJD"
        my_cmap = copy.copy(mpl.cm.get_cmap("CMRmap"))
        # my_cmap = copy.copy(sns.dark_palette("#A0e5f7", as_cmap=True))

    else:
        phase = phase / resolution

        size_time = np.arange(phase.min(), phase.max(), 1)
        # size_time = np.concatenate([size_time])#, [phase.max()]])
        size_time = size_time - int(phase.min())
        # print(len(size_time))
        # print(size_time)

        master = np.zeros([len(size_time), len(hello)])
        data_positions = []

        for k in range(len(phase)):
            data_pos = int(phase[k]) - int(phase.min())
            data_positions.append(data_pos)

        pos_breakdown = list(set(data_positions))
        for final_pos in pos_breakdown:
            pos = np.where(np.array(data_positions) == final_pos)[0]
            flx_pos = np.sum(np.array(supes)[pos], axis=0) / len(pos)

            master[final_pos] = flx_pos

        s_master = np.zeros([100, len(hello)])
        for cc in range(0, 100):
            phase_range = [0.01 * cc, 0.01 * cc + 0.05]
            # ic(phase_range)
            s_mask = np.ma.masked_inside(
                phase_keep, phase_range[0], phase_range[1]
            ).mask
            # ic(phase_keep[s_mask])
            s_flx_pos = np.sum(np.array(supes)[s_mask], axis=0) / len(
                np.array(supes)[s_mask]
            )
            if len(s_flx_pos) != len(hello):
                s_flx_pos = np.zeros(len(hello))
            s_master[cc] = s_flx_pos

        # print(len(master))
        # 2 phases
        master = np.tile(s_master, (2, 1))

        set_time_limits = [2, 0]
        set_time_label = "Phase (P = {:.2f})".format(P)
        my_cmap = copy.copy(sns.dark_palette("#A0e5f7", as_cmap=True))

    # my_cmap = copy.copy(sns.dark_palette("#A0e5f7", as_cmap=True))
    # my_cmap = copy.copy(sns.color_palette("bone", as_cmap=True))
    # print(np.log10(master))
    if log_scale:
        masked_array = np.ma.masked_where(master == 0, np.log10(master))
    else:
        masked_array = np.ma.masked_where(master == 0, master)
    # print(masked_array)

    #
    my_cmap.set_bad(color="white")

    if save_plot:
        fig = plt.figure(1, figsize=(4, 8))
        gs1 = gridspec.GridSpec(4, 1, height_ratios=[0.05, 0.15, 1, 0.2])
        gs1.update(hspace=0.00, wspace=0.025)  # , top=0.9)
        ax = plt.subplot(gs1[2, 0])
        ax2 = plt.subplot(gs1[3, 0])
    else:
        ax = axes[0]
        ax2 = axes[1]

    # cbax = fig.add_axes([.85, 0.25, 0.03, 0.5])
    print(set_time_limits)
    if set_limits:
        img1 = ax.imshow(
            masked_array,
            cmap=my_cmap,
            interpolation="nearest",
            extent=[velmin, velmax, set_time_limits[0], set_time_limits[1]],
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
    else:
        img1 = ax.imshow(
            masked_array,
            cmap=my_cmap,
            interpolation="nearest",
            extent=[velmin, velmax, set_time_limits[0], set_time_limits[1]],
            aspect="auto",
        )

    ax_divider = make_axes_locatable(ax)
    # add an axes above the main axes.
    cbax = ax_divider.append_axes("top", size="5%", pad="10%")
    # cb = colorbar(im2, cax=cax2, orientation="horizontal")
    cb = Colorbar(ax=cbax, mappable=img1, orientation="horizontal", ticklocation="top")
    cb.set_label("Relative flux", fontsize=11)
    # cb.set_clim(-0.1, 0.1)
    # ax_cmax = plt.axes([0.15, 0.94, 0.65, 0.03])
    # ax_cmin = plt.axes([0.15, 0.90, 0.65, 0.03])
    # c_max = 2
    # c_min = -2
    #
    # s_cmax = Slider(ax_cmax, "max", -2, 3, valfmt=c_max)
    # s_cmin = Slider(ax_cmin, "min", -2, 3, valfmt=c_min)
    #
    # def update(val, s=None):
    #     _cmin = s_cmin.val
    #     _cmax = s_cmax.val
    #     ic([_cmin, _cmax])
    #     img1.set_clim([_cmin, _cmax])
    #     plt.draw()
    #
    # s_cmax.on_changed(update)
    # s_cmin.on_changed(update)

    plt.setp(ax.get_xticklabels(), visible=False)
    # ax.set_title(line)
    # t = np.linspace(0., 2., 100)
    # t = np.linspace(MJD_all.min(), MJD_all.max(), 1000)

    # THIS WAS FOR GCAS HALPHA
    if not phase_folded and subplot == "last":
        # t = np.linspace(t0, MJD_all.max(), 1000)
        # sine = np.sin(2 * np.pi / P * t)
        # ax.set_ylim(MJD_keep.max(), MJD_keep.min())
        # V/R max 1 = RJD = 2512
        # V/R min 1 = RJD 2827
        # V/R max 2 = RJD 3133
        # V/R min 2 = RJD 3287
        # V/R max 3 = 3398
        # V/R min 3 = 3513
        # V/R max 4 = 3605
        ax.annotate(
            "",
            xy=(velmax, 2512),
            xytext=(velmax + velmax / 5.0, 2512),
            arrowprops=dict(arrowstyle="-|>", alpha=0.5, fc="b", ec="b"),
        )
        ax.annotate(
            r"$V/R_{MAX}$",
            xy=(velmax, 2512),
            xycoords="data",
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
        ax.annotate(
            "",
            xy=(velmax, 2827),
            xytext=(velmax + velmax / 5.0, 2827),
            arrowprops=dict(arrowstyle="-|>", alpha=0.5, fc="r", ec="r"),
        )
        ax.annotate(
            r"$V/R_{MIN}$",
            xy=(velmax, 2827),
            xycoords="data",
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

        ax.annotate(
            "",
            xy=(velmax, 3133),
            xytext=(velmax + velmax / 5.0, 3133),
            arrowprops=dict(arrowstyle="-|>", alpha=0.5, fc="b", ec="b"),
        )
        # ax.annotate(r'$V/R_{MAX}$', xy=(velmax, 3138),xycoords='data',
        #        xytext=(5, 5), textcoords='offset points',fontsize=8)
        ax.annotate(
            "",
            xy=(velmax, 3287),
            xytext=(velmax + velmax / 5.0, 3287),
            arrowprops=dict(arrowstyle="-|>", alpha=0.5, fc="r", ec="r"),
        )
        # ax.annotate(r'$V/R_{MIN}$', xy=(velmax, 3287), xycoords='data',
        #        xytext=(5, 5), textcoords='offset points',fontsize=8)

        ax.annotate(
            "",
            xy=(velmax, 3398),
            xytext=(velmax + velmax / 5.0, 3398),
            arrowprops=dict(arrowstyle="-|>", alpha=0.5, fc="b", ec="b"),
        )
        ax.annotate(
            "",
            xy=(velmax, 3513),
            xytext=(velmax + velmax / 5.0, 3513),
            arrowprops=dict(arrowstyle="-|>", alpha=0.5, fc="r", ec="r"),
        )
        ax.annotate(
            "",
            xy=(velmax, 3605),
            xytext=(velmax + velmax / 5.0, 3605),
            arrowprops=dict(arrowstyle="-|>", alpha=0.5, fc="b", ec="b"),
        )
        # ax.annotate(r'$V/R_{MAX}$', xy=(velmax, 3381),xycoords='data',
        #        xytext=(5, 5), textcoords='offset points',fontsize=8)
        # black_line = np.loadtxt("Ha_peak_RVs_fit.txt").T
        # ax.plot(black_line[1], black_line[0], "k", lw=1)

    else:
        t = np.linspace(0.0, 2.0, 100)
        sine = np.sin(np.linspace(np.pi / 2, 4.5 * np.pi, 100))
        amp = 55.9
        # ax.plot(amp * sine, t, color="xkcd:darkish red", lw=0.6)

    if subplot == "first":
        ax2.set_ylabel("Norm. flux", fontsize=9)
        ax.set_ylabel(set_time_label)
    else:
        # plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)

    ax2.set_xlabel(r"$\mathrm{Velocity\,[km\,s^{-1}]}$")
    nbins = 4
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8, prune="upper"))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune="upper"))

    # new_color = plt.cm.jet(phase_keep)

    for flxs in flx_all:
        ax2.plot(temp, flxs, color="gray", alpha=0.3, lw=0.5)

    ax2.plot(temp, hello, color=my_cmap(0.25), lw=2)
    ax2.set_xlim(velmin, velmax)
    # ax2.set_ylim(0.95, 2.2)

    # ax.axvline(-200, ls=':', color='k', lw=0.5)
    # ax.axvline(200, ls=':', color='k', lw=0.5)
    # ax2.axvline(-200, ls=':', color='k', lw=0.5)
    # ax2.axvline(200, ls=':', color='k', lw=0.5)
    # ax.yaxis.grid(False) # Hide the horizontal gridlines
    # ax2.yaxis.grid(False) # Hide the horizontal gridlines
    # ax.xaxis.grid(True) # Show the vertical gridlines
    # ax2.xaxis.grid(True) # Show the vertical gridlines
    date = dt.datetime.today().strftime("%d-%m-%y")

    # plt.show()
    # plt.show()

    if set_limits and save_plot:
        plt.savefig(
            line + "_dynamic_" + date + "_" + str(vmin) + "_" + str(vmax) + ".pdf",
            dpi=100,
            bbox_inches="tight",
        )
    elif save_plot:
        plt.savefig(line + "_dynamic_" + date + ".pdf", dpi=100, bbox_inches="tight")

    # temp2 = np.arange(-850, +850, 0.2)
    ##
    # plt.figure(2, figsize=(4,8))
    # ax1 = plt.subplot(111)
    #
    # for k in range(len(master)):
    #    #for l in range(len(MJD_a)):
    #    plt.scatter(temp, master[k]+MJD_all[k], c=master[k], cmap=my_cmap, s=0.5)
    # ax1.set_ylim(ax1.get_ylim()[::-1])
    # plt.savefig(line+"_dynamic_profiles.png", dpi=100, bbox_inches='tight')
