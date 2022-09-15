# These functions calculate strain in amorphous samples.

import numpy as np
from py4DSTEM.process.calibration import fit_ellipse_amorphous_ring
from py4DSTEM.process.utils import tqdmnd

def fit_amorphous_halo(
    datacube,
    center,
    fitradii,
    # return_ellipse_5_params = False,
    ):
    '''
    Loop through all probe positions and fit amorphous halo.

    Args:
        datacube (DataCube):            py4DSTEM datacube
        center (numpy.array):           (x,y) center coordinate guess in pixels
        fitradii (numpy.array):         (radius_inner,radius_outer) radial fitting range in pixels
        return_ellipse_5_params (bool): Set to true to return both 11- and 5-param ellipse coefficients


    Returns:
        ellipse_params_11 (numpy.array):    11-param ellipse coefficients for each probe position
        ellipse_params_5 (numpy.array):     (optional) 5-param ellipse coefficients 
    
    '''

    # Convert inputs to numpy arrays
    center = np.array(center)
    fitradii = np.array(fitradii)
    
    # Fit ellipse to mean diffraction pattern
    datacube.get_dp_mean()
    _, p11 = fit_ellipse_amorphous_ring(
        datacube.tree['dp_mean'].data, 
        center, 
        fitradii, 
    )
    # update center guess
    center = (p11[6],p11[7])

    # init
    # ellipse_params_5 = np.zeros((
    #     dataset.data.shape[0],
    #     dataset.data.shape[1],
    #     5,
    # ))
    ellipse_params_11 = np.zeros((
        datacube.data.shape[0],
        datacube.data.shape[1],
        11,
    ))
    
    # main loop
    for rx,ry in tqdmnd(datacube.data.shape[0],datacube.data.shape[1]):
    #     for rx,ry in py4DSTEM.process.utils.tqdmnd(2,2):
        _, p11_fit = fit_ellipse_amorphous_ring(
            datacube.data[rx,ry],
            center, 
            fitradii, 
            p0 = p11,
        )
        # ellipse_params_5[rx,ry,:] = p5_fit
        ellipse_params_11[rx,ry,:] = p11_fit

    
    return ellipse_params_11
     # ellipse_params_5
