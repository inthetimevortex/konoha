import numpy as np
import pyhdust.spectools as spt


def find_bisectors(
    vl_ar, flx_ar, vlrange, counter_val=1.5, emission_frac=0.5, region_size_flx=0.3
):
    """Jon's routine"""

    flx_ar = flx_ar[((vl_ar > -vlrange) & (vl_ar < vlrange))]
    vl_ar = vl_ar[((vl_ar > -vlrange) & (vl_ar < vlrange))]
    max_em_val = np.max(flx_ar)
    bis_vals = []
    bis_flx_vals = []
    bis_V_RVs = []
    bis_R_RVs = []
    # while counter_val < emission_frac * (1.0 + max_em_val):
    while counter_val < (emission_frac * (max_em_val - 1) + 1.0):
        cond_Ha = (flx_ar > counter_val) & (
            flx_ar < counter_val + 1.0 * region_size_flx
        )
        vel_slice = vl_ar[cond_Ha]
        flx_slice = flx_ar[cond_Ha]
        counter_val += region_size_flx

        bin_slice_V = np.mean(vel_slice[vel_slice < 0])
        bin_slice_R = np.mean(vel_slice[vel_slice > 0])
        bis_slice = np.mean([bin_slice_V, bin_slice_R])

        # bis_slice = np.mean(vel_slice)
        bis_V_RVs.append(bin_slice_V)
        bis_R_RVs.append(bin_slice_R)
        bis_vals.append(bis_slice)
        bis_flx_vals.append(np.mean(flx_slice))

    mean_bis_val = np.mean(np.array(bis_vals)[np.logical_not(np.isnan(bis_vals))])
    EW_V = spt.EWcalc(
        vl_ar[((vl_ar < mean_bis_val))] + mean_bis_val,
        flx_ar[((vl_ar < mean_bis_val))],
        vw=vlrange,
    )
    EW_R = spt.EWcalc(
        vl_ar[((vl_ar > mean_bis_val))] + mean_bis_val,
        flx_ar[((vl_ar > mean_bis_val))],
        vw=vlrange,
    )
    # print(EW_V/EW_R)
    try:
        # print(EW_V/EW_R)
        return EW_V, EW_R, mean_bis_val
    except ZeroDivisionError:
        # EW_V/EW_R = 0
        return 0, mean_bis_val
