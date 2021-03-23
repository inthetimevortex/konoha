import pandas as pd
import numpy as np
import xdrlib
from mpl_toolkits.mplot3d import Axes3D
#df = pd.read_table("part.1.00.dat", sep="\s+")
#f0 = open('test4.xdr', 'wb')
#data = df.values
#helper = data.flatten()
#p = xdrlib.Packer()
##data2 = p.pack_farray(len(helper), helper, p.pack_float)
#x = p.pack_farray(len(data[0]), data[0], p.pack_float)
#y = p.pack_farray(len(data[0]), data[0], p.pack_float)
#z = p.pack_farray(len(data[0]), data[0], p.pack_float)
#rho = p.pack_farray(len(data[0]), data[0], p.pack_float)
#
#f0.write(p.get_buffer())
#f0.close()

def xdr_reader(fname):
    f = open(fname, 'rb').read()
    #f = open('pleione/data.000_vdd.xdr', 'rb').read()

    data5 = xdrlib.Unpacker(f)
    R_d = data5.unpack_double()
    nr = data5.unpack_int()

    rgrid = []
    for i in range(0, nr+1):
        rval =  data5.unpack_double()
        rgrid.append(rval)

    f1 = open('mu.txt','w')
    nmu = data5.unpack_int()
    f1.writelines('{0}\n'.format(nmu))
    mugrid = np.zeros([nr, nmu+1])
    for i in range(0, nr):
        for j in range(0, nmu+1):
            muval =  data5.unpack_double()
            f1.writelines('{0}\n'.format(muval))
            mugrid[i,j] = muval
    f1.close()


    f1 = open('phi.txt','w')
    nphi = data5.unpack_int()
    f1.writelines('{0}\n'.format(nphi))
    phigrid = []
    for i in range(0, nphi+1):
        phival =  data5.unpack_double()
        f1.writelines('{0}\n'.format(phival))
        phigrid.append(phival)
    f1.close()

    #f1 = open('rho.txt','w')
    rhogrid = np.zeros([nr, nmu, nphi])
    for i in range(0, nr):
        for j in range(0, nmu):
            for k in range(0, nphi):
                rhoval =  data5.unpack_float()
                rhogrid[i,j,k] = rhoval

    return R_d, rgrid, mugrid, phigrid, rhogrid


def xdr_maker(fname, Rd, nr, rgrid, nmu, mugrid, nphi, phigrid, xrho):
    p = xdrlib.Packer()
    p.pack_double(Rd)
    p.pack_int(nr)
    for i in range(0, nr+1):
        p.pack_double(rgrid[i])
    p.pack_int(nmu)
    for i in range(0, nr):
        for j in range(0, nmu+1):
            p.pack_double(mugrid[i,j])
    p.pack_int(nphi)
    for i in range(0, nphi+1):
        p.pack_double(phigrid[i])
    z = np.zeros([nr, nmu])
    for i in range(0, nr):
    #const_rho[i, 20:30, :] = 2.14e11
        for j in range(0, nmu):
            #z[i, j] = mutoz(mugrid[i, j], rgrid[i])
            for k in range(0, nphi):
                #if j ==0 or j == nmu-1:
                #    rhogrid[i, j, k] = 0.0
                #else:
                #    rhogrid[i, j, k], H[i], sigmagrid[i, k] = xrho
                p.pack_float(xrho[i,j,k])


    f0 = open(fname, 'wb')
    f0.write(p.get_buffer())
    f0.close()
    print('XDR FILE ' + fname+' CREATED')

###################################################################
###
###f = open('vdd.xdr', 'rb').read()
###data5 = xdrlib.Unpacker(f)
###R_d2 = data5.unpack_double()
###nr2 = data5.unpack_int()
###
###rgrid2 = []
###for i in range(0, nr2+1):
###    rval =  data5.unpack_double()
###    rgrid2.append(rval)
###
####f1 = open('mu.txt','w')
###nmu2 = data5.unpack_int()
####f1.writelines('{0}\n'.format(nmu))
###mugrid2 = np.zeros([nr2, nmu2+1])
###for i in range(0, nr):
###    for j in range(0, nmu+1):
###        muval =  data5.unpack_double()
###        #f1.writelines('{0}\n'.format(muval))
###        mugrid2[i,j] = muval
####f1.close()
###
###
####f1 = open('phi.txt','w')
###nphi2 = data5.unpack_int()
####f1.writelines('{0}\n'.format(nphi))
###phigrid2 = []
###for i in range(0, nphi2+1):
###    phival =  data5.unpack_double()
###    #f1.writelines('{0}\n'.format(phival))
###    phigrid2.append(phival)
####f1.close()
###
###
####f1 = open('rho.txt','w')
###rhogrid2 = np.zeros([nr2, nmu2, nphi2])
###for i in range(0, nr2):
###    for j in range(0, nmu2):
###        for k in range(0, nphi2):
###            rhoval =  data5.unpack_float()
###            rhogrid2[i,j,k] = rhoval
