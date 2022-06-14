#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  __init__.py
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

from __future__ import division, print_function, absolute_import
from .get_data import (
    get_halpha,
    get_iue,
    get_lines,
    get_sed,
    plot_lines,
    gen_plot_lines,
)
from .find_bisectors import find_bisectors
from .radial_velocity import delta_v, fwhm2sigma, gauss, Ha_delta_v
from .sed_tools import jy2cgs, color_corr, vosa2catvalues
from .utils import (
    gentkdates,
    kde_scipy,
    integral,
    griddataBA,
    griddataBA_new,
    griddataBAtlas,
    bin_data,
    find_nearest,
    find_neighbours,
    geneva_interp_fast,
    poly_interp,
    Sliding_Outlier_Removal,
)
from .xdr_reader import xdr_reader, xdr_maker, xdr_reader_vel
from .constants import *
from .sph_functions import (
    atoi,
    natural_keys,
    part_info,
    rotate_part,
    rotate_sec_part,
    kernel,
    calc_energy_part,
    kloop,
    get_particles,
    calc_loop,
    plot_sigma4alpha,
    density_energy,
)
from .linesDict import line_names
from .dynamic import dynamic_spectra
from .be_theory import t_tms_from_Xc, hfrac2tms, oblat2w, obl2W, W2oblat
from .colormaper import hex_to_rgb, rgb_to_dec, get_continuous_cmap


__version__ = "0.1"
__all__ = (
    "get_halpha",
    "get_iue",
    "get_lines",
    "get_sed",
    "find_bisectors",
    "plot_lines",
    "delta_v",
    "Ha_delta_v",
    "fwhm2sigma",
    "gauss",
    "jy2cgs",
    "color_corr",
    "vosa2catvalues",
    "kde_scipy",
    "integral",
    "griddataBA",
    "griddataBA_new",
    "griddataBAtlas",
    "bin_data",
    "find_nearest",
    "find_neighbours",
    "geneva_interp_fast",
    "poly_interp",
    "xdr_reader",
    "xdr_maker",
    "line_names",
    "dynamic_spectra",
    "gen_plot_lines",
    "Sliding_Outlier_Removal",
    "gentkdates",
    "t_tms_from_Xc",
    "hfrac2tms",
    "oblat2w",
    "obl2W",
    "W2oblat",
    "hex_to_rgb",
    "rgb_to_dec",
    "get_continuous_cmap",
)
