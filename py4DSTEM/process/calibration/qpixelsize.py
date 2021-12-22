# Functions for calibrating the pixel size in the diffraction plane.

import numpy as np
from scipy.optimize import leastsq
from typing import Union, Optional
from py4DSTEM.process.utils import get_CoM, tqdmnd
from ...io.datastructure import Calibrations, PointListArray


def get_Q_pixel_size(q_meas, q_known, units='A'):
    """
    Computes the size of the Q-space pixels.

    Args:
        q_meas (number): a measured distance in q-space in pixels
        q_known (number): the corresponding known *real space* distance
        unit (str): the units of the real space value of `q_known`

    Returns:
        (number,str): the detector pixel size, the associated units
    """
    return 1. / (q_meas * q_known), units+'^-1'


def get_dq_from_indexed_peaks(qs, hkl, a):
    """
    Get dq, the size of the detector pixels in the diffraction plane, in inverse length
    units, using a set of measured peak distances from the optic axis, their Miller
    indices, and the known unit cell size.

    Args:
        qs (array): the measured peak positions
        hkl (list/tuple of length-3 tuples): the Miller indices of the peak positions qs.
            The length of qs and hkl must be the same.  To ignore any peaks, for this
            peak set (h,k,l)=(0,0,0).
        a (number): the unit cell size

    Returns:
        (4-tuple): A 4-tuple containing:

            * **dq**: *(number)* the detector pixel size
            * **qs_fit**: *(array)* the fit positions of the peaks
            * **hkl_fit**: *(list/tuple of length-3 tuples)* the Miller indices of the
              fit peaks
            * **mask**: *(array of bools)* False wherever hkl[i]==(0,0,0)
    """
    assert len(qs) == len(hkl), "qs and hkl must have same length"

    # Get spacings
    d_inv = np.array([np.sqrt(a ** 2 + b ** 2 + c ** 2) for (a, b, c) in hkl])
    mask = d_inv != 0

    # Get scaling factor
    c0 = np.average(qs[mask] / d_inv[mask])
    fiterr = lambda c: qs[mask] - c * d_inv[mask]
    popt, _ = leastsq(fiterr, c0)
    c = popt[0]

    # Get pixel size
    dq = 1 / (c * a)
    qs_fit = d_inv[mask] / a
    hkl_fit = [hkl[i] for i in range(len(hkl)) if mask[i] == True]

    return dq, qs_fit, hkl_fit


def calibrate_Bragg_peaks_pixel_size(
    braggpeaks: PointListArray,
    q_pixel_size: Optional[float] = None,
    calibrations: Optional[Calibrations] = None,
    name: Optional[str] = None,
) -> PointListArray:
    """
    Calibrate reciprocal space measurements of Bragg peak positions, using
    either `q_pixel_size` or the `Q_pixel_size` field of a
    Calibrations object

    Accepts:
        braggpeaks (PointListArray) the detected, unscaled bragg peaks
        q_pixel_size (float) Q pixel size in inverse Ångström
        calibrations (Calibrations) an object containing pixel size
        name        (str, optional) a name for the returned PointListArray.
                    If unspecified, takes the old PLA name, removes '_raw'
                    if present at the end of the string, then appends
                    '_calibrated'.

    Returns:
        braggpeaks_calibrated  (PointListArray) the calibrated Bragg peaks
    """
    assert isinstance(braggpeaks, PointListArray)
    assert (q_pixel_size is not None) != (
        calibrations is not None
    ), "Either (qx0,qy0) or calibrations must be specified"

    if calibrations is not None:
        assert isinstance(calibrations, Calibrations), "calibrations must be a Calibrations object."
        q_pixel_size = calibrations.get_Q_pixel_size()
        assert q_pixel_size is not None, "calibrations did not contain center position"

    if q_pixel_size is not None:
        assert isinstance(q_pixel_size, float), "q_pixel_size must be a float."

    if name is None:
        sl = braggpeaks.name.split("_")
        _name = "_".join(
            [s for i, s in enumerate(sl) if not (s == "raw" and i == len(sl) - 1)]
        )
        name = _name + "_calibrated"
    assert isinstance(name, str)

    braggpeaks_calibrated = braggpeaks.copy(name=name)

    for Rx, Ry in tqdmnd(
        braggpeaks_calibrated.shape[0], braggpeaks_calibrated.shape[1]
    ):
        pointlist = braggpeaks_calibrated.get_pointlist(Rx, Ry)
        pointlist.data["qx"] *= q_pixel_size
        pointlist.data["qy"] *= q_pixel_size

    return braggpeaks_calibrated
