import numpy as np
import scipy.special as sp
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


# Compute spherical harmonic pattern
def sph_harm(l, m):

    lon = np.linspace(0,2*np.pi,200)-np.pi
    lat = np.linspace(-np.pi/2,np.pi/2,500)
    colat = lat+np.pi/2
    d = np.zeros((len(lon),len(colat)),dtype = np.complex64)
    for j, yy in enumerate(colat):
        for i, xx in enumerate(lon):
            d[i,j] = sp.sph_harm(m,l,xx,yy)
    drm = np.transpose(np.real(d)) #only interested in real components
    return lon, lat, drm

#Set up figure

fig = plt.figure(figsize=(8,8))
gs = fig.add_gridspec(4, 4)
inclination = 60 #degrees from pole
plotcrs = ccrs.Orthographic(20, 90 - inclination)
ls = [0, 1, 2, 3]
ms = [0, 1, 2, 3]

for l in ls:
    for m in ms:
        if m <= l:
            ax = fig.add_subplot(gs[l,m], projection=plotcrs)
            lon, lat, drm = sph_harm(l, m)
            #Plot, limiting colors to extreme data values
            vlim = np.max(np.abs(drm))
            ax.pcolormesh(lon*180/np.pi,lat*180/np.pi,drm,transform=ccrs.PlateCarree(),
                                    cmap='jet',vmin=-vlim,vmax=vlim)
            #ax.set_xlabel('m = {}'.format(m))
            #ax.set_ylabel('l = {}'.format(l))
            if m == 0:
                ax.text(-0.30, 0.40, 'l = {}'.format(l), va='bottom', ha='center',
                    rotation='horizontal', rotation_mode='anchor',
                    transform=ax.transAxes, fontsize=13)
            if l == 3:
                ax.text(0.5, -0.25, 'm = {}'.format(m), va='bottom', ha='center',
                    rotation='horizontal', rotation_mode='anchor',
                    transform=ax.transAxes, fontsize=13)


#Necessary function calls


plt.savefig('lm.png',dpi=300)
