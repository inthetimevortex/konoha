import numpy as np
from scipy.interpolate import griddata
import pyhdust.phc as phc
from scipy.stats import gaussian_kde
from scipy.signal import detrend
from astropy.stats import median_absolute_deviation as MAD
from astropy.convolution import convolve, Box1DKernel
import jdcal
import datetime as dt
import struct as struct
from scipy.interpolate import UnivariateSpline


def readpck(n, tp, ixdr, f):
    """ Read XDR

    - n: length
    - tp: type ('i', 'l', 'f', 'd')
    - ixdr: counter
    - f: file-object

    :returns: ixdr (counter), np.array
    """
    sz = dict(zip(["i", "l", "f", "d"], [4, 4, 4, 8]))
    s = sz[tp]
    upck = ">{0}{1}".format(n, tp)
    return ixdr + n * s, np.array(struct.unpack(upck, f[ixdr : ixdr + n * s]))


def readXDRsed(xdrpath, quiet=False):
    """  Read a XDR with a set of models.

    The models' parameters (as well as their units) are defined at XDR
    creation.

    INPUT: xdrpath

    OUTPUT: ninfo, intervals, lbdarr, minfo, models

    (xdr dimensions, params limits, lambda array (um), mods params, mods flux)
    """
    ixdr = 0
    f = open(xdrpath, "rb").read()
    ixdr, ninfo = readpck(3, "l", ixdr, f)
    nq, nlbd, nm = ninfo
    ixdr, intervals = readpck(nq * 2, "f", ixdr, f)
    ixdr, lbdarr = readpck(nlbd, "f", ixdr, f)
    ixdr, listpar = readpck(nq * nm, "f", ixdr, f)
    ixdr, models = readpck(nlbd * nm, "f", ixdr, f)
    #
    if ixdr == len(f):
        if not quiet:
            print("# XDR {0} completely read!".format(xdrpath))
    else:
        _warn.warn(
            "# XDR {0} not completely read!\n# length "
            "difference is {1} /4".format(xdrpath),
            (len(f) - ixdr),
        )
    #
    return (
        ninfo,
        intervals.reshape((nq, 2)),
        lbdarr,
        listpar.reshape((nm, nq)),
        models.reshape((nm, nlbd)),
    )


def readBAsed(xdrpath, quiet=False):
    """ Read **only** the BeAtlas SED release.

    | Definitions:
    | -photospheric models: sig0 (and other quantities) == 0.00
    | -Parametric disk model default (`param` == True)
    | -VDD-ST models: n excluded (alpha and R0 fixed. Confirm?)
    | -The models flux are given in ergs/s/cm2/um. If ignorelum==True in the
    |   XDR creation, F_lbda/F_bol unit will be given.

    INPUT: xdrpath

    | OUTPUT: listpar, lbdarr, minfo, models
    | (list of mods parameters, lambda array (um), mods index, mods flux)
    """
    with open(xdrpath, "rb") as fi:
        f = fi.read()
        ixdr = 0
        #
        npxs = 3
        upck = ">{0}l".format(npxs)
        header = np.array(struct.unpack(upck, f[ixdr : ixdr + npxs * 4]))
        ixdr += npxs * 4
        nq, nlb, nm = header
        #
        npxs = nq
        upck = ">{0}l".format(npxs)
        header = np.array(struct.unpack(upck, f[ixdr : ixdr + npxs * 4]))
        ixdr += npxs * 4
        #
        listpar = [[] for i in range(nq)]
        for i in range(nq):
            npxs = header[i]
            upck = ">{0}f".format(npxs)
            listpar[i] = np.array(struct.unpack(upck, f[ixdr : ixdr + npxs * 4]))
            ixdr += npxs * 4
        #
        npxs = nlb
        upck = ">{0}f".format(npxs)
        lbdarr = np.array(struct.unpack(upck, f[ixdr : ixdr + npxs * 4]))
        ixdr += npxs * 4
        #
        npxs = nm * (nq + nlb)
        upck = ">{0}f".format(npxs)
        models = np.array(struct.unpack(upck, f[ixdr : ixdr + npxs * 4]))
        ixdr += npxs * 4
        models = models.reshape((nm, -1))
        # this will check if the XDR is finished.
        if ixdr == len(f):
            if not quiet:
                print("# XDR {0} completely read!".format(xdrpath))
        else:
            _warn.warn(
                "# XDR {0} not completely read!\n# length "
                "difference is {1}".format(xdrpath, (len(f) - ixdr) / 4)
            )
        #
        # f.close()
    return listpar, lbdarr, models[:, 0:nq], models[:, nq:]


def sigma_spectra(vel, flux, mask):
    """ Uses the std of the residue of the continuum fit as
    a way to estimate sigma for spectra"""

    spl_weight = np.zeros(len(vel))
    spl_weight[mask] = 1.0
    spl_fit = UnivariateSpline(vel, flux, w=spl_weight, k=3)
    flx_norm = flux / spl_fit(vel)

    residue = flux[mask] - spl_fit(vel)[mask]
    sigma = np.std(residue)
    return sigma


def gentkdates(mjd0, mjd1, fact, step, dtstart=None):
    """ Generates round dates between > mjd0 and < mjd1 in a given step.
    Valid steps are:

        'd/D/dd/DD' for days;
        'm/M/mm/MM' for months;
        'y/Y/yy/YY/yyyy/YYYY' for years.

    dtstart (optional) is expected to be in datetime.datetime.date() format
    [i.e., datetime.date(yyyy, m, d)]

    fact must be an integer
    """

    # check sanity of dtstart
    if dtstart is None:
        dtstart = dt.datetime(*jdcal.jd2gcal(jdcal.MJD_0, mjd0)[:3]).date()
        mjdst = jdcal.gcal2jd(dtstart.year, dtstart.month, dtstart.day)[1]
    else:
        mjdst = jdcal.gcal2jd(dtstart.year, dtstart.month, dtstart.day)[1]
        if mjdst < mjd0 - 1 or mjdst > mjd1:
            print('# Warning! Invalid "dtstart". Using mjd0.')
            dtstart = dt.datetime(*jdcal.jd2gcal(jdcal.MJD_0, mjd0)[:3]).date()
    # define step 'position' and vector:
    basedata = [dtstart.year, dtstart.month, dtstart.day]
    dates = []
    mjd = mjdst
    if step.upper() in ["Y", "YY", "YYYY"]:
        i = 0
        while mjd < mjd1 + 1:
            dates += [dt.datetime(*basedata).date()]
            basedata[i] += fact
            mjd = jdcal.gcal2jd(*basedata)[1]
    elif step.upper() in ["M", "MM"]:
        i = 1
        while mjd < mjd1 + 1:
            dates += [dt.datetime(*basedata).date()]
            basedata[i] += fact
            while basedata[i] > 12:
                basedata[0] += 1
                basedata[1] -= 12
            mjd = jdcal.gcal2jd(*basedata)[1]
    elif step.upper() in ["D", "DD"]:
        i = 2
        daysvec = np.arange(1, 29, fact)
        if basedata[i] not in daysvec:
            j = 0
            while daysvec[j + 1] < basedata[i]:
                j += 1
            daysvec += basedata[i] - daysvec[j]
            idx = np.where(daysvec < 29)
            daysvec = daysvec[idx]
        else:
            j = np.where(daysvec == basedata[i])[0]
        while mjd < mjd1 + 1:
            dates += [dt.datetime(*basedata).date()]
            j += 1
            if j == len(daysvec):
                j = 0
                basedata[1] += 1
                if basedata[1] == 13:
                    basedata[1] = 1
                    basedata[0] += 1
            basedata[i] = daysvec[j]
            mjd = jdcal.gcal2jd(*basedata)[1]
    else:
        print("# ERROR! Invalid step")
        raise SystemExit(1)
    return dates


def Sliding_Outlier_Removal(x, y, window_size, sigma=3.0, iterate=1, replace_pts=False):
    # remove NANs from the data
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]

    # make sure that the arrays are in order according to the x-axis
    y = y[np.argsort(x)]
    x = x[np.argsort(x)]

    index_of_all_clipped = np.array([])

    window_size_original = window_size

    # tells you the difference between the last and first x-value
    x_span = x.max() - x.min()
    i = 0
    x_final = x
    y_final = y
    while i < iterate:
        i += 1
        x = x_final
        y = y_final

        # empty arrays that I will append not-clipped data points to
        x_good_ = np.array([])
        y_good_ = np.array([])

        # used to calculate the average standard deviation in the bins in each iteration
        std_of_bins = []

        # Creates an array with all_entries = True. index where you want to remove outliers are set to False
        tf_ar = np.full((len(x),), True, dtype=bool)
        ar_of_index_of_bad_pts = np.array([])  # not used anymore

        # this is how many days (or rather, whatever units x is in) to slide the window center when finding the outliers
        slide_by = window_size / 5.0

        # calculates the total number of windows that will be evaluated
        Nbins = int((int(x.max() + 1) - int(x.min())) / slide_by)

        if i == 1:
            box_jitter = 0
        else:
            box_jitter = np.random.uniform(-0.125, 0.125)
            box_jitter = 0
            window_size = window_size * (1 - 0.01 * i)

        if window_size / window_size_original < 0.2:
            window_size = window_size_original * 0.2
        # print('window size = {0}'.format(window_size))

        for j in range(Nbins + 1):

            # find the minimum time in this bin, and the maximum time in this bin
            x_bin_min = (
                x.min() + j * (slide_by) - 0.5 * window_size + box_jitter * window_size
            )
            x_bin_max = (
                x.min() + j * (slide_by) + 0.5 * window_size + box_jitter * window_size
            )

            # gives you just the data points in the window
            x_in_window = x[(x > x_bin_min) & (x < x_bin_max)]
            y_in_window = y[(x > x_bin_min) & (x < x_bin_max)]

            # if there are less than 5 points in the window, do not try to remove outliers.
            if len(y_in_window) > 5:

                # Removes a linear trend from the y-data that is in the window.
                y_detrended = detrend(y_in_window, type="linear")
                y_in_window = y_detrended

                # Records the standard deviation in the data in the box after the linear fit is subtracted
                std_of_bins.append(np.std(y_in_window))

                # print(np.median(m_in_window_))
                y_med = np.median(y_in_window)

                # finds the Median Absolute Deviation of the y-pts in the window
                y_MAD = MAD(y_in_window)

                # This mask returns the not-clipped data points.
                # Maybe it is better to only keep track of the data points that should be clipped...
                mask_a = (y_in_window < y_med + y_MAD * sigma) & (
                    y_in_window > y_med - y_MAD * sigma
                )
                # print(str(np.sum(mask_a)) + '   :   ' + str(len(m_in_window)))
                y_good = y_in_window[mask_a]
                x_good = x_in_window[mask_a]

                y_bad = y_in_window[~mask_a]
                x_bad = x_in_window[~mask_a]

                # keep track of the index --IN THE ORIGINAL FULL DATA ARRAY-- of pts to be clipped out
                try:
                    clipped_index = np.where([x == z for z in x_bad])[1]
                    tf_ar[clipped_index] = False
                    ar_of_index_of_bad_pts = np.concatenate(
                        [ar_of_index_of_bad_pts, clipped_index]
                    )
                except IndexError:
                    # print('no data between {0} - {1}'.format(x_in_window.min(), x_in_window.max()))
                    pass
            # puts the 'good' not-clipped data points into an array to be saved

            # x_good_= np.concatenate([x_good_, x_good])
            # y_good_= np.concatenate([y_good_, y_good])

            # print(len(mask_a))
            # print(len(m
            # print(m_MAD)

        ##multiple data points will be repeated! We don't want this, so only keep unique values.
        # x_uniq, x_u_indexs = np.unique(x_good_, return_index=True)
        # y_uniq = y_good_[x_u_indexs]

        ar_of_index_of_bad_pts = np.unique(ar_of_index_of_bad_pts)
        # print('step {0}: remove {1} points. Mean STD = {2:.5f}'.format(i, len(ar_of_index_of_bad_pts), np.mean(std_of_bins)))
        index_of_all_clipped = np.concatenate(
            (index_of_all_clipped, ar_of_index_of_bad_pts)
        )
        # print(ar_of_index_of_bad_pts)

        # x_bad = x[ar_of_index_of_bad_pts]
        # y_bad = y[ar_of_index_of_bad_pts]
        # x_final = x[

        x_final = x[tf_ar]
        y_final = y[tf_ar]

    # trying to replace the clipped points with something sensible, rather than simply removing them
    index_of_all_clipped = np.unique(index_of_all_clipped)

    return (x_final, y_final, index_of_all_clipped)


# ==============================================================================
def kde_scipy(x, x_grid, bandwidth=0.2):
    """Kernel Density Estimation with Scipy"""

    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1))
    return kde.evaluate(x_grid)


# ==============================================================================
def integral(x, f, cummulative=False):
    """
    Integration using trapezoidal rule

    Usage:
    integ = integral(x, f, cummulative=False)
    """
    x = np.array(x).reshape((-1))
    f = np.array(f).reshape((-1))
    if cummulative:
        integ = np.array([np.trapz(f[0:i], x=x[0:i]) for i in range(len(x))])
    else:
        integ = np.trapz(f, x=x)

    return integ


# ==============================================================================


def poly_interp(xi, yi, x):
    """
    For N pair of points, interpolates a polynomial
    of (N-1) order

    xi, yi: original data points
    x , y : interpolated values (x is an input)

    Usage:
    y = poly_interp(xi, yi, x)
    """
    # make sure they are numpy arrays
    xi = np.array([xi]).reshape((-1))
    yi = np.array([yi]).reshape((-1))
    x = np.array([x]).reshape((-1))

    # definitions
    n = len(xi)
    y = 0.0

    # loop
    for i in range(n):
        num = yi[i]
        den = 1.0
        for j in range(n):
            if j != i:
                num = num * (x - xi[j])
                den = den * (xi[i] - xi[j])

        y = y + num / den

    return y


# ==============================================================================
# BIN DATA
def bin_data(x, y, nbins, xran=None, exclude_empty=True):
    """
    Bins data

    Usage:
    xbin, ybin, dybin = bin_data(x, y, nbins, xran=None, exclude_empty=True)

    where dybin is the standard deviation inside the bins.
    """
    # make sure it is a numpy array
    x = np.array([x]).reshape((-1))
    y = np.array([y]).reshape((-1))
    # make sure it is in increasing order
    ordem = x.argsort()
    x = x[ordem]
    y = y[ordem]

    if xran is None:
        xmin, xmax = x.min(), x.max()
    else:
        xmin, xmax = xran[0], xran[1]

    xborders = np.linspace(xmin, xmax, nbins + 1)
    xbin = 0.5 * (xborders[:-1] + xborders[1:])

    ybin = np.zeros(nbins)
    dybin = np.zeros(nbins)
    for i in range(nbins):
        aux = (x > xborders[i]) * (x < xborders[i + 1])
        if np.array([aux]).any():
            ybin[i] = np.mean(y[aux])
            dybin[i] = np.std(y[aux])
        else:
            ybin[i] = np.nan
            dybin[i] = np.nan

    if exclude_empty:
        keep = np.logical_not(np.isnan(ybin))
        xbin, ybin, dybin = xbin[keep], ybin[keep], dybin[keep]

    return xbin, ybin, dybin


# ==============================================================================
def find_nearest(array, value):
    """
    Find the nearest value inside an array
    """

    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


# ==============================================================================
def find_neighbours(par, par_grid, ranges):
    """
    Finds neighbours' positions of par in par_grid.

    Usage:
    keep, out, inside_ranges, par_new, par_grid_new = \
        find_neighbours(par, par_grid, ranges):

    where redundant columns in 'new' values are excluded,
    but length is preserved (i.e., par_grid[keep] in griddata call).
    """
    # check if inside ranges

    if len(par) == 4:
        ranges = ranges[0:4]
    if len(par) == 3:
        ranges = ranges[0:3]
    # print(par, len(ranges))
    # print(par, ranges)
    count = 0
    inside_ranges = True
    while (inside_ranges is True) * (count < len(par)):
        inside_ranges = (par[count] >= ranges[count, 0]) * (
            par[count] <= ranges[count, 1]
        )
        count += 1

    # find neighbours
    keep = np.array(len(par_grid) * [True])
    out = []

    if inside_ranges:
        for i in range(len(par)):
            # coincidence
            if (par[i] == par_grid[:, i]).any():
                keep *= par[i] == par_grid[:, i]
                out.append(i)
            # is inside
            else:
                # list of values
                par_list = np.array(list(set(par_grid[:, i])))
                # nearest value at left
                par_left = par_list[par_list < par[i]]
                par_left = par_left[np.abs(par_left - par[i]).argmin()]
                # nearest value at right
                par_right = par_list[par_list > par[i]]
                par_right = par_right[np.abs(par_right - par[i]).argmin()]
                # select rows
                kl = par_grid[:, i] == par_left
                kr = par_grid[:, i] == par_right
                keep *= kl + kr
        # delete coincidences
        par_new = np.delete(par, out)
        par_grid_new = np.delete(par_grid, out, axis=1)
    else:
        print("Warning: parameter outside ranges.")
        par_new = par
        par_grid_new = par_grid

    return keep, out, inside_ranges, par_new, par_grid_new


# ==============================================================================
def geneva_interp_fast_old(Par, oblat, t, neighbours_only=True, isRpole=False):
    """
    Interpolates Geneva stellar models, from grid of
    pre-computed interpolations.

    Usage:
    Rpole, logL = geneva_interp_fast(Mstar, oblat, t,
                                     neighbours_only=True, isRpole=False)
    or
    Mstar, logL = geneva_interp_fast(Rpole, oblat, t,
                                     neighbours_only=True, isRpole=True)
    (in this case, the option 'neighbours_only' will be set to 'False')

    where t is given in tMS, and tar is the open tar file. For now, only
    Z=0.014 is available.
    """
    # from my_routines import find_neighbours
    from scipy.interpolate import griddata

    # read grid
    dir0 = "~/Dropbox/Amanda/GRID/defs/geneve_models/"
    fname = "geneva_interp_Z014.npz"
    data = np.load(dir0 + fname)
    Mstar_arr = data["Mstar_arr"]
    oblat_arr = data["oblat_arr"]
    t_arr = data["t_arr"]
    Rpole_grid = data["Rpole_grid"]
    logL_grid = data["logL_grid"]

    # build grid of parameters
    par_grid = []
    for M in Mstar_arr:
        for ob in oblat_arr:
            for tt in t_arr:
                par_grid.append([M, ob, tt])
    par_grid = np.array(par_grid)

    # set input/output parameters
    if isRpole:
        Rpole = Par
        par = np.array([Rpole, oblat, t])
        Mstar_arr = par_grid[:, 0].copy()
        par_grid[:, 0] = Rpole_grid.flatten()
        neighbours_only = False
    else:
        Mstar = Par
        par = np.array([Mstar, oblat, t])
    # print(par)

    # set ranges
    ranges = np.array(
        [[par_grid[:, i].min(), par_grid[:, i].max()] for i in range(len(par))]
    )

    # find neighbours
    if neighbours_only:
        keep, out, inside_ranges, par, par_grid = find_neighbours(par, par_grid, ranges)
    else:
        keep = np.array(len(par_grid) * [True])
        # out = []
        # check if inside ranges
        count = 0
        inside_ranges = True
        while (inside_ranges is True) * (count < len(par)):
            inside_ranges = (par[count] >= ranges[count, 0]) * (
                par[count] <= ranges[count, 1]
            )
            count += 1

    # interpolation method
    if inside_ranges:
        interp_method = "linear"
    else:
        print("Warning: parameters out of available range," + " taking closest model.")
        interp_method = "nearest"

    if len(keep[keep]) == 1:
        # coincidence
        if isRpole:
            Mstar = Mstar_arr[keep][0]
            Par_out = Mstar
        else:
            Rpole = Rpole_grid.flatten()[keep][0]
            Par_out = Rpole
        logL = logL_grid.flatten()[keep][0]
    else:
        # interpolation
        if isRpole:
            Mstar = griddata(
                par_grid[keep], Mstar_arr[keep], par, method=interp_method, rescale=True
            )[0]
            Par_out = Mstar
        else:
            Rpole = griddata(
                par_grid[keep],
                Rpole_grid.flatten()[keep],
                par,
                method=interp_method,
                rescale=True,
            )[0]
            Par_out = Rpole
        logL = griddata(
            par_grid[keep],
            logL_grid.flatten()[keep],
            par,
            method=interp_method,
            rescale=True,
        )[0]

    return Par_out, logL


############################################################################
def geneva_interp_fast(Mstar, oblat, t, Zstr="014", silent=True):
    """
    Interpolates Geneva stellar models, from grid of
    pre-computed interpolations.

    Usage:
    Rpole, logL, age = geneva_interp_fast(Mstar, oblat, t, Zstr='014')

    where t is given in tMS, and tar is the open tar file. For now, only
    Zstr='014' is available.
    """
    # read grid
    # dir0 = '{0}/refs/geneva_models/'.format(_hdtpath())
    dir0 = "/Users/arubio/Google Drive/Meu Drive/Dropbox/Amanda/GRID/BEMCEE/defs/geneve_models/"
    if Mstar <= 20.0:
        fname = "geneva_interp_Z{:}.npz".format(Zstr)
    else:
        fname = "geneva_interp_Z{:}_highM.npz".format(Zstr)
    data = np.load(dir0 + fname)
    Mstar_arr = data["Mstar_arr"]
    oblat_arr = data["oblat_arr"]
    t_arr = data["t_arr"]
    Rpole_grid = data["Rpole_grid"]
    logL_grid = data["logL_grid"]
    age_grid = data["age_grid"]

    # build grid of parameters
    par_grid = []
    for M in Mstar_arr:
        for ob in oblat_arr:
            for tt in t_arr:
                par_grid.append([M, ob, tt])
    par_grid = np.array(par_grid)

    # set input/output parameters
    par = np.array([Mstar, oblat, t])

    # set ranges
    ranges = np.array(
        [[par_grid[:, i].min(), par_grid[:, i].max()] for i in range(len(par))]
    )

    # find neighbours
    keep, out, inside_ranges, par, par_grid = find_neighbours(par, par_grid, ranges)

    # interpolation method
    if inside_ranges:
        interp_method = "linear"
    else:
        if not silent:
            print(
                "[geneva_interp_fast] Warning: parameters out of available range, taking closest model"
            )
        interp_method = "nearest"

    if len(keep[keep]) == 1:
        # coincidence
        Rpole = Rpole_grid.flatten()[keep][0]
        logL = logL_grid.flatten()[keep][0]
        age = age_grid.flatten()[keep][0]
    else:
        # interpolation
        Rpole = griddata(
            par_grid[keep],
            Rpole_grid.flatten()[keep],
            par,
            method=interp_method,
            rescale=True,
        )[0]
        logL = griddata(
            par_grid[keep],
            logL_grid.flatten()[keep],
            par,
            method=interp_method,
            rescale=True,
        )[0]
        age = griddata(
            par_grid[keep],
            age_grid.flatten()[keep],
            par,
            method=interp_method,
            rescale=True,
        )[0]

    return Rpole, logL, age


# ===============================================================================


def griddataBA(minfo, models, params, listpar, dims):
    """
    Moser's routine to interpolate BeAtlas models
    obs: last argument ('listpar') had to be included here
    """

    # print(params[0])
    idx = np.arange(len(minfo))
    lim_vals = len(params) * [
        [],
    ]
    for i in range(len(params)):
        # print(i, listpar[i], params[i], minfo[:, i])
        lim_vals[i] = [
            phc.find_nearest(listpar[i], params[i], bigger=False),
            phc.find_nearest(listpar[i], params[i], bigger=True),
        ]
        tmp = np.where(
            (minfo[:, i] == lim_vals[i][0]) | (minfo[:, i] == lim_vals[i][1])
        )
        idx = np.intersect1d(idx, tmp[0])

    out_interp = griddata(minfo[idx], models[idx], params)[0]

    if np.sum(out_interp) == 0 or np.sum(np.isnan(out_interp)) > 0:

        mdist = np.zeros(np.shape(minfo))
        ichk = range(len(params))
        for i in ichk:
            mdist[:, i] = np.abs(minfo[:, i] - params[i]) / (
                np.max(listpar[i]) - np.min(listpar[i])
            )
        idx = np.where(np.sum(mdist, axis=1) == np.min(np.sum(mdist, axis=1)))
        if len(idx[0]) != 1:
            out_interp = griddata(minfo[idx], models[idx], params)[0]
        else:
            out_interp = models[idx][0]

    # if (np.sum(out_interp) == 0 or np.sum(np.isnan(out_interp)) > 0) or\
    #    bool(np.isnan(np.sum(out_interp))) is True:
    #     print("# Houve um problema na grade e eu nao consegui arrumar...")

    return out_interp


# ==============================================================================
def hfrac2tms(Hfrac, inverse=False):
    """
    Converts nuclear hydrogen fraction into fractional time in the main-sequence,
    (and vice-versa) based on the polynomial fit of the average of this relation
    for all B spectral types and rotational velocities.

    Usage:
    t = hfrac2tms(Hfrac, inverse=False)
    or
    Hfrac = hfrac2tms(t, inverse=True)
    """
    if not inverse:
        coef = np.array([-0.57245754, -0.8041484, -0.51897195, 1.00130795])
        tms = coef.dot(np.array([Hfrac ** 3, Hfrac ** 2, Hfrac ** 1, Hfrac ** 0]))
    else:
        # interchanged parameter names
        coef = np.array([-0.74740597, 0.98208541, -0.64318363, -0.29771094, 0.71507214])
        tms = coef.dot(
            np.array([Hfrac ** 4, Hfrac ** 3, Hfrac ** 2, Hfrac ** 1, Hfrac ** 0])
        )

    return tms


# ==============================================================================
def griddataBAtlas(minfo, models, params, listpar, dims, isig):
    idx = range(len(minfo))
    lim_vals = len(params) * [
        [],
    ]
    for i in [i for i in range(len(params)) if i != isig]:
        lim_vals[i] = [
            phc.find_nearest(listpar[i], params[i], bigger=False),
            phc.find_nearest(listpar[i], params[i], bigger=True),
        ]
        tmp = np.where(
            (minfo[:, i] == lim_vals[i][0]) | (minfo[:, i] == lim_vals[i][1])
        )
        idx = np.intersect1d(idx, tmp)
        #
    out_interp = griddata(minfo[idx], models[idx], params)[0]
    #
    if np.sum(out_interp) == 0 or np.sum(np.isnan(out_interp)) > 0:
        print("# Houve um problema na grade. Tentando arrumadar...")
        idx = np.arange(len(minfo))
        for i in [i for i in range(len(params)) if i != dims["sig0"]]:
            imin = lim_vals[i][0]
            if lim_vals[i][0] != np.min(listpar[i]):
                imin = phc.find_nearest(listpar[i], lim_vals[i][0], bigger=False)
            imax = lim_vals[i][1]
            if lim_vals[i][1] != np.max(listpar[i]):
                imax = phc.find_nearest(listpar[i], lim_vals[i][1], bigger=True)
            lim_vals[i] = [imin, imax]
            tmp = np.where(
                (minfo[:, i] >= lim_vals[i][0]) & (minfo[:, i] <= lim_vals[i][1])
            )
            idx = np.intersect1d(idx, phc.flatten(tmp))
        out_interp = griddata(minfo[idx], models[idx], params)[0]
    #
    if np.sum(out_interp) == 0 or np.sum(np.isnan(out_interp)) > 0:
        print("# Houve um problema na grade e eu nao conseguir arrumar...")
    #
    return out_interp


# ==============================================================================
def griddataBA_new(minfo, models, params, isig, silent=True):
    """
    Interpolates model grid

    Usage:
    model_interp = griddata(minfo, models, params, isig, silent=True)

    where
    minfo = grid of parameters
    models = grid of models
    params = parameters,
    isig = (normalized) sigma0 index

    Ex:
    # read grid
    xdrpath = 'beatlas/disk_flx.xdr'
    listpar, lbdarr, minfo, models = bat.readBAsed(xdrpath, quiet=True)
    # find isig
    dims = ['M', 'ob', 'sig0', 'nr', 'cosi']
    dims = dict(zip(dims, range(len(dims))))
    isig = dims["sig0"]
    # interpolation
    params = [12.4, 1.44, 0.9, 4.4, 0.1]
    model_interp = np.exp(griddataBA(minfo, np.log(models), params, isig))

    If photospheric models are interpolated, let isig=None. For spectra,
    it is recommended to enter the log of the grid of spectra as input,
    as shown in the example above.
    """
    # ranges
    ranges = np.array([[parr.min(), parr.max()] for parr in minfo.T])

    # find neighbours, delete coincidences
    if phc.is_inside_ranges(isig, [0, len(params) - 1]):
        # exclude sig0 dimension, to take all their entries for interpolation
        keep, out, inside_ranges, params1, minfo1 = phc.find_neighbours(
            np.delete(params, isig),
            np.delete(minfo, isig, axis=1),
            np.delete(ranges.T, isig, axis=1).T,
            silent=silent,
        )
        params = np.hstack([params1, params[isig]])
        minfo = np.vstack([minfo1.T, minfo[:, isig]]).T
    else:
        keep, out, inside_ranges, params, minfo = phc.find_neighbours(
            params, minfo, ranges, silent=silent
        )

    # interpolation
    model_interp = griddata(minfo[keep], models[keep], params, method="linear")[0]

    if np.isnan(model_interp).any() or np.sum(model_interp) == 0.0:
        if not silent:
            print(
                "[griddataBA] Warning: linear interpolation didnt work, taking closest model"
            )
        model_interp = griddata(minfo[keep], models[keep], params, method="nearest")[0]

    return model_interp
