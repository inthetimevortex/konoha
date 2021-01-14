#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  bynary_calculator.py
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


import numpy as np
from utils import geneva_interp_fast

M1 = 4.89
M2 =4.5
Msun = 1.98e33 #g
Rsun= 6.96e10 #cm
G = 6.67e-8 #cgs
M = (M1+M2)*Msun
W=0.90
ob = W**2 / 2 + 1
t1 = 0.85
Raio, logL= geneva_interp_fast(M1, ob, t1)   
Req = Raio * ob
Rtrunc = Req*19.77*Rsun
a = Rtrunc/0.8

P = 2*np.pi*np.sqrt(a**3/(G*M))
Pday = P/(60*60*24)

i= np.deg2rad(39) 
M2 = Msun* M2

K3 = (M2**3 * np.sin(i)**3)/M**2 * (2*np.pi*G)/P
K = K3**(1/3) *10**-5 #km/s
