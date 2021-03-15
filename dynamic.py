#!/usr/bin/env python
# -*- coding: utf-8 -*-
'Dynamic Spectra maker!!'
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
import pandas as pd
from scipy.interpolate import griddata
from matplotlib.colorbar import Colorbar
import seaborn as sns
import copy
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

sns.set_style("ticks", {"xtick.major.direction": 'in',
              "ytick.major.direction": 'in'})





#vel_all, flux_all, MJD_all, flag_all = get_halpha()
def dynamic_spectra(line, MJD_all, vel_all, flux_all, resolution, vmin, vmax, set_limits, phase_folded, P, t0):

    #resolution = 0.5 #days
    velmin, velmax = -450, 450
    flx_all=[]
    temp = np.arange(velmin, velmax, 1)


    MJD_to_sort = np.array(MJD_all)
    sort = MJD_to_sort.argsort()
    MJD_all = np.array(MJD_to_sort)[sort]
    #flag_all = np.array(flag_all)[sort]
    vel_all = np.array(vel_all,dtype=object)[sort]
    flux_all = np.array(flux_all,dtype=object)[sort]
    #print(len(MJD_all))
    #print(vel_all)
    #print(flux_all)
    for i in range(len(flux_all)):
        # interpolates specrum on 'temp' so they all have the same size :)
        flx = griddata(vel_all[i], flux_all[i], temp, method='linear')
        #plt.plot(vel_all[i], flux_all[i], linewidth=0.4, label=MJD_all[i])
        flx_all.append(flx)
    #print(flx_all)
    flxx = np.array(flx_all)
    keep = np.logical_not(np.isnan(np.array(flx_all)))[:,0]
    flx_all = flxx[keep]
    vel_all = vel_all[keep]
    MJD_all = MJD_all[keep]
    MJD_to_sort = np.array(MJD_all)
    sort = MJD_to_sort.argsort()
    #plt.legend()
    #plt.show()
    #print(len(MJD_all))
    #im so good at names
    hello = np.mean(flx_all, axis=0)

    supes = []


    for j in range(len(flx_all)):
        #superflux = np.tile(flx_all[j]-hello, (1, 1))
        superflux = flx_all[j]-hello
        #superflux = np.tile(flx_all[j], (1, 1))
        #supes.append((superflux+ 1)**3 - 1)
        supes.append(superflux)



    #flux_a = np.tile(flx_all[0], (1, 1))
    MJD_a = MJD_all[sort]
    #print(len(MJD_a), len(MJD_all))
    MJD_a = MJD_a - min(MJD_a)
    phase = (MJD_a-t0)/P %1
    phase_keep = np.copy(phase)


    if not phase_folded:
        MJD_a = MJD_a/resolution

        size_time = np.arange(MJD_a.min(), MJD_a.max(), 1)
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
            flx_pos = np.sum(np.array(supes)[pos], axis=0)/len(pos)

            master[final_pos] = flx_pos

        set_time_limits = [MJD_all.max(), MJD_all.min()]
        set_time_label = 'MJD'

    else:
        phase = phase/resolution

        size_time = np.arange(phase.min(), phase.max(), 1)
        #size_time = np.concatenate([size_time])#, [phase.max()]])
        size_time = size_time - int(phase.min())
        print(len(size_time))
        print(size_time)

        master = np.zeros([len(size_time), len(hello)])
        data_positions = []

        for k in range(len(phase)):
            data_pos = int(phase[k])- int(phase.min())
            data_positions.append(data_pos)

        pos_breakdown = list(set(data_positions))
        for final_pos in pos_breakdown:
            pos = np.where(np.array(data_positions) == final_pos)[0]
            flx_pos = np.sum(np.array(supes)[pos], axis=0)/len(pos)

            master[final_pos] = flx_pos
        print(len(master))
        master = np.tile(master, (2,1))
        print(len(master))
        set_time_limits = [2, 0]
        set_time_label = 'Phase (P = {:.2f})'.format(P)



    masked_array = np.ma.masked_where(master == 0, master)
    #my_cmap = mpl.cm.viridis
    my_cmap = copy.copy(mpl.cm.get_cmap("viridis"))
    my_cmap.set_bad(color='white')



    fig = plt.figure(1, figsize=(4, 8))
    gs1 = gridspec.GridSpec(4, 1, height_ratios=[0.05,0.15,1,0.2])
    gs1.update(hspace=0.00, wspace=0.025)#, top=0.9)


    #fig.subplots_adjust(right=0.8)



    ax = plt.subplot(gs1[2, 0])
    #cbax = fig.add_axes([.85, 0.25, 0.03, 0.5])
    if set_limits:
        img1 = ax.imshow(masked_array, cmap = my_cmap,interpolation='nearest', extent = [velmin, velmax, set_time_limits[0], set_time_limits[1]],aspect='auto' , vmin=vmin, vmax=vmax)
    else:
        img1 = ax.imshow(masked_array, cmap = my_cmap,interpolation='nearest', extent = [velmin, velmax, set_time_limits[0], set_time_limits[1]],aspect='auto')

    ax_divider = make_axes_locatable(ax)
    # add an axes above the main axes.
    cbax = ax_divider.append_axes("top", size="5%", pad="2%")
    #cb = colorbar(im2, cax=cax2, orientation="horizontal")
    cb = Colorbar(ax = cbax, mappable = img1, orientation = 'horizontal', ticklocation = 'top')
    cb.set_label('Relative flux', fontsize=11)
    #cb.set_clim(-0.1, 0.1)

    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_ylabel(set_time_label)
    #ax.set_title(line)
    t = np.linspace(0., 2., 100)
    X2 = np.linspace(np.pi/2, 4.5*np.pi, 100)
    sine = np.sin(X2)
    amp=55.9

    ax.plot(amp*sine, t, color='xkcd:darkish red', lw=0.6)
    #amp=4.
    #ax.plot(amp*sine, t, color='xkcd:strawberry', lw=0.6)


    #temp2 = np.arange(-850, +850, 0.2)
    #
    #ax1 = plt.subplot(gs1[2, 1])
    #
    #
    #for k in range(len(master)):
    #    #for l in range(len(MJD_a)):
    #    plt.scatter(temp, 4*master[k]+MJD_all[k], c=master[k], cmap=my_cmap, s=0.5)
    #ax1.set_ylim(ax1.get_ylim()[::-1])


    ax2 = plt.subplot(gs1[3, 0])
    ax2.set_ylabel('Flux')#, fontsize=13)
    ax2.set_xlabel('$\mathrm{Velocity\,[km\,s^{-1}]}$')
    nbins = 4
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8, prune='upper'))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper'))
    hello = np.mean(flx_all, axis=0)

    #new_color = plt.cm.jet(phase_keep)
    #print(new_color)

    for flxs in flx_all:
        ax2.plot(temp, flxs, color='gray', alpha=0.3, lw=0.5)

    ax2.plot(temp, hello, color = my_cmap(.25), lw=2)
    ax2.set_xlim(velmin, velmax)

    ax.axvline(-200, ls=':', color='k', lw=0.5)
    ax.axvline(200, ls=':', color='k', lw=0.5)
    ax2.axvline(-200, ls=':', color='k', lw=0.5)
    ax2.axvline(200, ls=':', color='k', lw=0.5)
    #ax.yaxis.grid(False) # Hide the horizontal gridlines
    #ax2.yaxis.grid(False) # Hide the horizontal gridlines
    #ax.xaxis.grid(True) # Show the vertical gridlines
    #ax2.xaxis.grid(True) # Show the vertical gridlines

    plt.savefig(line+"_dynamic.pdf", dpi=100, bbox_inches='tight')


    #temp2 = np.arange(-850, +850, 0.2)
    ##
    #plt.figure(2, figsize=(4,8))
    #ax1 = plt.subplot(111)
    #
    #for k in range(len(master)):
    #    #for l in range(len(MJD_a)):
    #    plt.scatter(temp, master[k]+MJD_all[k], c=master[k], cmap=my_cmap, s=0.5)
    #ax1.set_ylim(ax1.get_ylim()[::-1])
    #plt.savefig(line+"_dynamic_profiles.png", dpi=100, bbox_inches='tight')
