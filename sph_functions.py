import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import re
from konoha import constants as cons
from joblib import Parallel, delayed
import warnings
from glob import glob

warnings.filterwarnings("error")


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def part_info(fname):
    """
    Reads info from part.xxx.dat type files.
    Returns x, y, z, vz, h, mass, rho, r, phi, vr, vphi
    velocities are in km/s (get_output.c l151-155)
    positions are in Rstar
    """
    df = pd.read_table(fname, sep="\s+").values
    x, y, z, vx, vy, vz, h, mass, rho = (
        df[:, 0],
        df[:, 1],
        df[:, 2],
        df[:, 3],
        df[:, 4],
        df[:, 5],
        df[:, 6],
        df[:, 7],
        df[:, 8],
    )
    # polar coords
    r = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    vx = vx * 1e5  # km/s to cm/s
    vy = vy * 1e5
    vz = vz * 1e5
    vphi = vy * np.cos(phi) - vx * np.sin(phi)
    vr = vy * np.sin(phi) + vx * np.cos(phi)

    return x, y, z, vx, vy, vz, h, mass, rho, r, phi, vr, vphi


def rotate_part(fname):
    x1, y1, z, vx1, vy1, vz, h, mass, rho, r, phi, vr, vphi = part_info(fname)

    phi = np.arctan2(y1, x1)
    sec_pos = phi[1]
    phi = (phi - sec_pos + np.pi) % (2 * np.pi) - np.pi

    x = x1 * np.cos(-sec_pos) - y1 * np.sin(-sec_pos)
    y = x1 * np.sin(-sec_pos) + y1 * np.cos(-sec_pos)
    r = np.sqrt(x ** 2 + y ** 2)

    vx = vx1 * np.cos(-sec_pos) - vy1 * np.sin(-sec_pos)
    vy = vx1 * np.sin(-sec_pos) + vy1 * np.cos(-sec_pos)

    vr = vx * np.cos(phi) + vy * np.sin(phi)
    vphi = vy * np.cos(phi) - vx * np.sin(phi)

    return x, y, z, vx, vy, vz, h, mass, rho, r, phi, vr, vphi


def rotate_sec_part(fname):
    x1, y1, z, vx1, vy1, vz, h, mass, rho, r, phi, vr, vphi = part_info(fname)
    # i want the secondary (xyz[1]) to be in the center of the coord system
    x1 = x1 - x1[1]
    y1 = y1 - y1[1]
    z = z - z[1]

    vx1 = vx1 - vx1[1]
    vy1 = vy1 - vy1[1]
    vz = vz - vz[1]

    phi = np.arctan2(y1, x1)
    # this is now the primary
    pri_pos = phi[0]
    phi = (phi - pri_pos + np.pi) % (2 * np.pi) - np.pi

    x = x1 * np.cos(-pri_pos) - y1 * np.sin(-pri_pos)
    y = x1 * np.sin(-pri_pos) + y1 * np.cos(-pri_pos)
    r = np.sqrt(x ** 2 + y ** 2)

    vx = vx1 * np.cos(-pri_pos) - vy1 * np.sin(-pri_pos)
    vy = vx1 * np.sin(-pri_pos) + vy1 * np.cos(-pri_pos)

    vr = vx * np.cos(phi) + vy * np.sin(phi)
    vphi = vy * np.cos(phi) - vx * np.sin(phi)

    return x, y, z, vx, vy, vz, h, mass, rho, r, phi, vr, vphi


def kernel(r, h):
    """
    KERNEL FUNCTION

    """

    q = np.abs(r / h)  # //Both r and h should be positive
    w = 1.0 / np.pi / (h ** 3)
    if q < 1.0:
        w *= 1.0 - 3.0 / 2.0 * (q ** 2) + 3.0 / 4.0 * (q ** 3)
    elif q < 2.0:
        w *= 1.0 / 4.0 * (2.0 - q) ** 3
    else:
        w = 0.0
    return w


# def dkernel(r, h):
#     '''
#     dW/dq
#     '''
#     q = np.abs(r / h) # //Both r and h should be positive
#     gradw = 1.0 / np.pi / (h**3) / (r*h)
#
#     if (q < 1.):
#         gradw *= ( -3.*q + 2.25*(q*q))
#     elif (q < 2.):
#         gradw *= 3. * (q - 0.25*q*q - 1.)
#     else:
#         gradw = 0.
#
#     return gradw


def calc_energy_part(Rstar, x, y, z, vx, vy, vz, mass, r):
    """Calculates kinetic and potential energy of particles"""

    which = "s"

    ET = []  # cm3 g-1 s-2

    # center simulation on secondary
    r2 = np.sqrt(
        ((x - x[1]) * Rstar * cons.Rsun) ** 2 + ((y - y[1]) * Rstar * cons.Rsun) ** 2
    )

    M = mass[0] + mass[1]
    xcmass = 1.0 / M * (mass[0] * x[0] + mass[1] * x[1])
    ycmass = 1.0 / M * (mass[0] * y[0] + mass[1] * y[1])
    zcmass = 1.0 / M * (mass[0] * z[0] + mass[1] * z[1])
    rcmass = np.sqrt(xcmass ** 2 + ycmass ** 2 + zcmass ** 2)
    vxcmass = 1.0 / M * (mass[0] * vx[0] + mass[1] * vx[1])
    vycmass = 1.0 / M * (mass[0] * vy[0] + mass[1] * vy[1])
    vzcmass = 1.0 / M * (mass[0] * vz[0] + mass[1] * vz[1])

    for i in range(len(mass)):
        # ENERGIES RELATING TO THE PRIMARY
        try:
            Ec1 = 1 / 2 * mass[i] * (vx[i] ** 2 + vy[i] ** 2 + vz[i] ** 2)
            Ep1 = (-cons.G * mass[i] * mass[0]) / (r[i] * Rstar * cons.Rsun)
        except (ZeroDivisionError, RuntimeWarning):
            Ec1 = np.nan
            Ep1 = np.nan

        # ENERGIES RELATING TO THE SECONDARY
        vels = (vx[i] - vx[1]) ** 2 + (vy[i] - vy[1]) ** 2 + (vz[i] - vz[1]) ** 2
        Ec2 = 1 / 2 * mass[i] * vels
        try:  # EP in relation to the secondary
            Ep2 = (-cons.G * mass[i] * mass[1]) / r2[i]
        except RuntimeWarning:
            Ep2 = np.nan

        # ENERGIES RELATING TO CENTER OF MASS
        vels = (vx[i] - vxcmass) ** 2 + (vy[i] - vycmass) ** 2 + (vz[i] - vzcmass) ** 2
        Eccmass = 1 / 2 * mass[i] * vels
        try:
            Epcmass = (-cons.G * mass[i] * M) / ((r[i] - rcmass) * Rstar * cons.Rsun)
        except RuntimeWarning:
            Epcmass = np.nan
        Et1 = Ec1 + Ep1
        Et2 = Ec2 + Ep2
        Etcmass = Eccmass + Epcmass
        if which == "p":
            ET.append(Et1 * 1e-10)  # to kilo Joule
        if which == "s":
            ET.append(Et2 * 1e-10)  # to kilo Joule
        if which == "c":
            ET.append(Etcmass * 1e-10)  # to kilo Joule

    ET = np.array(ET)

    return ET


def kloop(fname, coords, follow2, Rstar, quantity, rhomean):

    dr, dphi, dz, centre = coords
    rhoCell, vrCell, energyCell, momentumCell = 0, 0, 0, 0

    if follow2:
        if "no_bi" in fname:  # isolated Be star simulation!
            x, y, z, vx, vy, vz, h, mass, rho, r, phi, vr, vphi = part_info(fname)
            size_opt = 1
        else:
            x, y, z, vx, vy, vz, h, mass, rho, r, phi, vr, vphi = rotate_part(fname)
            size_opt = 2
    else:
        x, y, z, vx, vy, vz, h, mass, rho, r, phi, vr, vphi = rotate_sec_part(fname)
        size_opt = 2

    if "energy" in quantity:
        ET = calc_energy(Rstar, x, y, z, vx, vy, vz, mass, r)
    # run through particles, excluding the stars
    for i in range(size_opt, len(r)):
        # is the particle inside the cell?
        if dr[0] < r[i] < dr[1] and dphi[0] < phi[i] < dphi[1] and dz[0] < z[i] < dz[1]:
            posx = np.abs(r[i] - centre[0])
            posy = np.abs(phi[i] - centre[1])
            posz = np.abs(z[i] - centre[2])

            pos = np.sqrt(posx ** 2 + posy ** 2 + posz ** 2)
            w = kernel(pos, h[i]) / (Rstar * cons.Rsun) ** 3  # kernel in cm-3

            if "rho" or "sigma" in quantity:
                rhoCell += mass[i] * w

            if "vr" in quantity:
                if not rhomean:
                    vrCell += mass[i] / rho[i] * vr[i] * w
                else:
                    vrCell += mass[i] / rho[i] * vr[i] * w * mass[i] * w
                # quantityCell.update({'vr':vrCell})
            if "energy" in quantity:
                energyCell += mass[i] / rho[i] * ET[i] * w
            if "momentum" in quantity:  # NOT TESTED, PROB WRONG AS HELL
                prod = (
                    y[i] * vz[i]
                    + z[i] * vx[i]
                    + x[i] * vy[i]
                    - z[i] * vy[i]
                    - x[i] * vz[i]
                    - y[i] * vx[i]
                )
                momentumCell += (
                    (mass[i] / cons.Msun) ** 2 / rho[i] * prod * cons.Rsun * Rstar * w
                )

    return rhoCell, vrCell, energyCell, momentumCell


def get_particles(
    coords,
    flist,
    Rstar,
    quantity=["sigma", "vr"],
    follow2=True,
    rhomean=True,
    ntsteps=35,
    pool=False,
):
    # read cell positions and part info
    # dr, dphi, dz, centre = coords # 2 2 2 3
    # rhoCell, vrCell, energyCell, momentumCell = 0, 0, 0, 0

    # create dictionary
    quantityCell = {}

    # ntsteps = 35
    if pool:
        # print("Imma loop!!!")
        result = Parallel(n_jobs=48)(
            delayed(kloop)(fname, coords, follow2, Rstar, quantity, rhomean)
            for fname in flist[-ntsteps:]
        )
        # ic(result)
        rhoCell, vrCell, energyCell, momentumCell = np.sum(result, axis=0)
        # ic(rhoCell)
    else:
        rhoCell, vrCell, energyCell, momentumCell = 0, 0, 0, 0
        for fname in flist[-ntsteps:]:
            u_rhoCell, u_vrCell, u_energyCell, u_momentumCell = kloop(
                fname, coords, follow2, Rstar, quantity, rhomean
            )
            rhoCell += u_rhoCell
            vrCell += u_vrCell
            energyCell += u_energyCell
            momentumCell += u_momentumCell

    # adds cell value to the dictionary
    quantityCell.update(
        {
            "rho": rhoCell / ntsteps,
            "sigma": rhoCell / ntsteps,
            "vr": vrCell / ntsteps,
            "energy": energyCell / ntsteps,
            "momentum": momentumCell / ntsteps,
        }
    )

    if rhomean:
        bvr = quantityCell.get("vr")
        brho = quantityCell.get("rho")
        try:
            quantityCell.update({"vr": bvr / brho})
        except (RuntimeWarning, ZeroDivisionError):
            quantityCell.update({"vr": np.nan})

    return quantityCell


def calc_loop(grid, flist, Rstar, quantity, **kwargs):
    rGrid, phiGrid, zGrid = grid

    if len(phiGrid) == 2:
        use_phi = False
    else:
        use_phi = True

    if len(zGrid) == 2:
        use_z = False
    else:
        use_z = True

    all_quants = {}
    # quantityFull = []
    # quantityFull = np.zeros([len(rGrid)-1, len(phiGrid)-1, len(zGrid)-1])
    if "sigma" in quantity:
        sigmaFull = np.zeros([len(rGrid) - 1, len(phiGrid) - 1])
        rhoFull = np.zeros([len(rGrid) - 1, len(phiGrid) - 1, len(zGrid) - 1])
        # all_quants.update({'sigma':sigmaFull})
    if "rho" in quantity:
        rhoFull = np.zeros([len(rGrid) - 1, len(phiGrid) - 1, len(zGrid) - 1])
        all_quants.update({"rho": rhoFull})
    if "vr" in quantity:
        vrFull = np.zeros([len(rGrid) - 1, len(phiGrid) - 1, len(zGrid) - 1])
        all_quants.update({"vr": vrFull})
    if "energy" in quantity:
        energyFull = np.zeros([len(rGrid) - 1, len(phiGrid) - 1, len(zGrid) - 1])
        all_quants.update({"energy": energyFull})
    if "momentum" in quantity:
        momentumFull = np.zeros([len(rGrid) - 1, len(phiGrid) - 1, len(zGrid) - 1])
        all_quants.update({"momentum": momentumFull})

    for i in range(len(rGrid) - 1):
        for j in range(len(phiGrid) - 1):
            for k in range(len(zGrid) - 1):
                dr = np.array([rGrid[i], rGrid[i + 1]])
                dphi = np.array([phiGrid[j], phiGrid[j + 1]])
                dz = np.array([zGrid[k], zGrid[k + 1]])
                centre = [np.mean(dr), np.mean(dphi), np.mean(dz)]

                holder = get_particles(
                    [dr, dphi, dz, centre], flist, Rstar, quantity, **kwargs
                )

                if "rho" or "sigma" in quantity:
                    rhoFull[i, j, k] = holder.get("rho")
                if "vr" in quantity:
                    vrFull[i, j, k] = holder.get("vr")
                if "energy" in quantity:
                    energyFull[i, j, k] = holder.get("energy")
                if "momentum" in quantity:
                    momentumFull[i, j, k] = holder.get("momentum")

                del_x3 = np.abs(np.diff(dz)) * Rstar * cons.Rsun
                if "sigma" in quantity:
                    # bsig = quantityFull[i,j,k].get('sigma')
                    sigmaFull[i, j] += rhoFull[i, j, k] * del_x3

            # quantityFull[i,j,k].update({'sigma':sigmaFull[i,j]})

    if not use_phi or use_z:
        for key in all_quants:
            all_quants[key] = np.reshape(all_quants[key], [len(rGrid) - 1])
        all_quants.update({"sigma": np.reshape(sigmaFull, [len(rGrid) - 1])})
    for key in all_quants:
        if use_z and not use_phi:
            all_quants[key] = np.reshape(
                all_quants[key], [len(rGrid) - 1, len(zGrid) - 1]
            )
        if use_phi and not use_z:
            all_quants[key] = np.reshape(
                all_quants[key], [len(rGrid) - 1, len(phiGrid) - 1]
            )

    return all_quants


def plot_sigma4alpha(alpha=0.1):

    if alpha == 0.1:
        alp = "_01"
    elif alpha == 0.5:
        alp = "_05"
    elif alpha == 1:
        alp = "_1"

    files = ["30", "50", "84", "100"]
    files = (f + alp for f in files)

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, fname in enumerate(files):
        flist = glob(fname + "/avrg*dat")
        flist.sort(key=natural_keys)
        print(flist[-1])
        r1, s1 = np.loadtxt(flist[-1]).T
        ax.plot(r1, s1 * alpha, label=fname.split("_")[0] + " days", zorder=100)

    nobi = glob("8no_binary/avrg*dat")
    nobi.sort(key=natural_keys)[-1]
    r2, s2 = np.loadtxt(nobi).T
    ax.plot(r1, s1 * alpha, "gray", label="Isolated Be star", ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.legend()
    ax.set_ylim(1e-7, 100)

    plt.savefig("images/" + alp + "sigma.png")

    return print("plot saved")


def density_energy(proj):
    fname = "quantities/" + proj + "/" + "0.05.npz"

    fig, ax = plt.subplots(figsize=[7, 5])

    data = np.load(fname)
    ax.plot(
        data["r"][data["energy"] > 0.0],
        data["sigma"][data["energy"] > 0.0],
        ".",
        color="gray",
        label="prim",
    )
    ax.plot(
        data["r"][data["energy"] < 0.0],
        data["sigma"][data["energy"] < 0.0],
        ".",
        color="orange",
        label="sec",
    )
    ax.set_ylim(1e-7, 100)
    ax.set_yscale("log")
    ax.legend()

    #   an (x, y) pair of relative coordinates (0 is left or bottom, 1 is right or top)
    axins = inset_axes(
        ax,
        width="70%",
        height="70%",
        axes_class=mpl.projections.polar.PolarAxes,
        bbox_to_anchor=[0.5, 0.5],
    )

    flist = glob(proj + "/part*dat")
    flist.sort(key=natural_keys)
    fname = flist[-1]

    x, y, z, vx, vy, vz, h, mass, rho, r, phi, vr, vphi = part_info(fname)
    Rstar = 5.5
    ET = calc_energy(Rstar, x, y, z, vx, vy, vz, mass, r)

    axins.plot(phi[ET > 0.0], r[ET > 0.0], ".", color="gray", ms=1, alpha=0.4)
    axins.plot(phi[ET < 0.0], r[ET < 0.0], ".", color="orange", ms=1, alpha=0.4)
    axins.plot(phi[0], r[0], "ko")
    axins.plot(phi[1], r[1], "k*")
    axins.set_ylim(0, 70)

    plt.savefig("images/DE_" + proj + ".png")

    return print("plot saved")
