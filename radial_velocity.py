#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  radial_velocity.py
#
#  Copyright 2020 Amanda Rubio <amanda.rubio@usp.br>
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
from glob import glob
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import griddata, interp1d
import matplotlib.pyplot as plt


def delta_v(vel, flux, line):
    """
    Calcula o shift na velocidade, baseado no centro ajustado p gaussiana
    """
    # coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)
    v_corte = np.zeros(2)
    if line == "Ha":
        i_corte = np.argsort(np.abs((flux - 1.0) - (flux.max() - 1.0) * 0.5))
    else:
        i_corte = np.argsort(np.abs((flux - 1.0) - (flux.min() - 1.0) * 0.5))

    # i_corte = np.argsort(np.abs((flux - 1.) - (flux.max() - 1.) * .5))
    v_corte[0] = vel[i_corte[0]]

    v_aux = vel[i_corte[1]]
    count = 2
    while v_corte[0] * v_aux > 0:
        v_aux = vel[i_corte[count]]
        count += 1
    v_corte[1] = v_aux
    v_corte.sort()
    delt_v = 10.0

    v_trechos = np.array(
        [v_corte[0] - delt_v, v_corte[0], v_corte[1], v_corte[1] + delt_v]
    )

    asas = np.where(
        (vel > v_trechos[0]) * (vel < v_trechos[1])
        + (vel > v_trechos[2]) * (vel < v_trechos[3])
    )

    plt.plot(vel[asas], flux[asas])
    plt.show()

    # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
    p0 = [flux.min() - 1.0, v_trechos.mean(), v_trechos[2] - v_trechos[1]]
    coeff, var_matrix = curve_fit(gauss, vel, flux - 1.0, p0=p0)

    return coeff[1]


def Ha_delta_v(vel, flux, line):

    deltav = delta_v(vel, flux, line)

    v_central = deltav

    n = len(vel)
    nn = (n - 1) / 2
    a = 150.0
    p1 = [flux.max() - 1.5, v_central - a, a / 3.5]
    p2 = [flux.max() - 1.5, v_central + a, a / 3.5]
    gaussian = gauss(vel, *p1) - gauss(vel, *p2)

    cc = []
    for i in range(0, 101):
        cv = np.sum(np.roll(gaussian, i - 50) * flux)
        cc.append(cv)
    xx = np.arange(0, 101)
    (g,) = np.where((np.array(cc) >= 0.0) & (xx >= 20.0) & (xx <= 80.0))

    vr = interp1d(
        cc[(g[0] - 1) : (g[0] + 1)], [v_central - 1.0, v_central + 1.0], kind="linear"
    )
    (vr,) = vr([0.0])

    return vr


def fwhm2sigma(fwhm):
    """Function to convert FWHM to standard deviation (sigma)"""
    return fwhm * np.sqrt(2.0) / (np.sqrt(2.0 * np.log(2.0)) * 2.0)


def gauss(x, *p):
    height, mu, fwhm = p
    return height * np.exp(-((x - mu) ** 2.0) / (2.0 * fwhm2sigma(fwhm) ** 2))
