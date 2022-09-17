# These functions calculate strain in amorphous samples.

import numpy as np
from py4DSTEM.process.calibration import fit_ellipse_amorphous_ring, double_sided_gaussian
from py4DSTEM.process.utils import tqdmnd
from py4DSTEM.io.datastructure import RealSlice
import matplotlib.pyplot as plt

import logging
import warnings
warnings.filterwarnings("ignore", message="Number of calls to function has reached maxfev")
warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")


def fit_amorphous_halo(
    datacube,
    center,
    fitradii,
    mask = None,
    fitbounds = True,
    plot_mean_ellipse_fit = False,
    ):
    '''
    Loop through all probe positions and fit amorphous halo.

    Args:
        datacube (DataCube):            py4DSTEM datacube
        center (numpy.array):           (x,y) center coordinate guess in pixels
        fitradii (numpy.array):         (radius_inner,radius_outer) radial fitting range in pixels
        mask (bool):                    Boolean mask - only pixels where mask = True will be used for fitting
        maxfev (int):                   Max number of function evals.  Set to <2400 to speed up fits.
        plot_mean_ellipse_fit (bool):   Show the ellipse fitting to the mean diffraction pattern.
                                        Note that if this option is true, we do not perform 
                                        ellipse fitting on the individual probe positions.

    Returns:
        ellipse_params_11 (numpy.array):    11-param ellipse coefficients for each probe position
    
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
        mask=mask,
        fitbounds=fitbounds,
    )
    print(np.round(p11[8:],decimals=6))
    # update center guess
    center = (p11[6],p11[7])

    # init
    ellipse_params_11 = np.zeros((
        datacube.data.shape[0],
        datacube.data.shape[1],
        11,
    ))

    # plotting
    if plot_mean_ellipse_fit:
        logging.warning(
            "\nElliptic fitting on all probe positions not performed." + \
            "\nSet plot_mean_ellipse_fit=False to perform fitting.",
            stacklevel=0,
            )

        fig,ax = plt.subplots(1,2,figsize=(16,8))

        yy,xx = np.meshgrid(
            np.arange(datacube.tree['dp_mean'].data.shape[1]),
            np.arange(datacube.tree['dp_mean'].data.shape[0]),
            )
        im_fit = double_sided_gaussian(
            p11,
            xx,
            yy,
            )
        if mask is not None:
            vmin = np.min(im_fit[mask])
            vmax = np.max(im_fit[mask])
        else:
            vmin = np.min(im_fit)
            vmax = np.max(im_fit)

        ax[0].imshow(
            datacube.tree['dp_mean'].data,
            vmin=vmin,
            vmax=vmax,
            cmap='turbo',
            )
        ax[0].set_title('Measured Mean Diffraction Pattern', fontsize=16)
        ax[1].imshow(
            im_fit,
            vmin=vmin,
            vmax=vmax,
            cmap='turbo',
            )
        ax[1].set_title('Fiting of Mean Diffraction Pattern', fontsize=16)
        plt.show()

    else:
        for rx,ry in tqdmnd(datacube.data.shape[0],datacube.data.shape[1]):
            _, p11_fit = fit_ellipse_amorphous_ring(
                datacube.data[rx,ry],
                center, 
                fitradii, 
                p0 = p11,
                mask=mask,
                fitbounds=fitbounds,
            )
            ellipse_params_11[rx,ry,:] = p11_fit

    
        return ellipse_params_11



def get_strain_amorphous(
    ellipse_params_11,
    ABC_ref = None,
    return_ref = False,
    print_ref = True,
    ):

    '''
    Strain calculation from amorphous halo fitting.


    Args:


    Returns:


    '''

    im_size = ellipse_params_11.shape[0:2]


    # Get reference ellipse from median if user does not provide it
    ABC_ref = np.median(ellipse_params_11[:,:,8:11],axis=(0,1))

    ### Get transformation matrix for reference ###
    # Transformation matrix is defined as the eigendecomposition of this ellipse matrix: 
    m_ellipse = np.array([[ABC_ref[0], ABC_ref[1]/2], [ABC_ref[1]/2, ABC_ref[2]]])
    e_vals, e_vecs = np.linalg.eig(m_ellipse)
    # Calculate the angle between the principal axes and original cartesian reference frame
    phi = np.arctan2(e_vecs[1, 0], e_vecs[0, 0])
    rot_matrix_ref = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    # The sqrt of the eigenvalues are the lengths of the major and minor axes
    transformation_matrix_ref = np.diag(np.sqrt(e_vals)) 
    # Rotate principle strains into original cartesian reference frame
    transformation_matrix_ref = rot_matrix_ref @ transformation_matrix_ref @ rot_matrix_ref.T

    if print_ref:
        print('reference (A,B,C) = ' + \
            f"({ABC_ref[0]:.6f},{ABC_ref[1]:.6f},{ABC_ref[2]:.6f})")


    # # Transformation matrix is defined as the eigendecomposition of this ellipse matrix: 
    # m_ellipse = np.array([[ABC_ref[0], ABC_ref[1]/2], [ABC_ref[1]/2, ABC_ref[2]]])
    # e_vals, e_vecs = np.linalg.eig(m_ellipse)
    # # Calculate the angle between the principal axes and original cartesian reference frame
    # phi = np.arctan2(e_vecs[1, 0], e_vecs[0, 0])
    # rot_matrix_ref = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    # # The sqrt of the eigenvalues are the lengths of the major and minor axes
    # transformation_matrix_ref = np.diag(np.sqrt(e_vals)) 
    # # Rotate principle strains into original cartesian reference frame
    # transformation_matrix_ref = rot_matrix_ref @ transformation_matrix_ref @ rot_matrix_ref.T

    # init
    strain_map = RealSlice(
        data=np.zeros((
            im_size[0],
            im_size[1],
            5)),
        slicelabels=('e_xx','e_yy','e_xy','theta','mask'),
        name='strain_map')
    strain_map.get_slice('mask').data[:] = 1.0

    # Loop over all probe positions
    for rx,ry in tqdmnd(im_size[0],im_size[1]):
        ABC_meas = ellipse_params_11[rx,ry,8:11]
        
        ### Get transformation matrix for measurement ###
        # Transformation matrix is defined as the eigendecomposition of this ellipse matrix: 
        m_ellipse_meas = np.array([[ABC_meas[0], ABC_meas[1]/2], [ABC_meas[1]/2, ABC_meas[2]]])
        e_vals, e_vecs = np.linalg.eig(m_ellipse_meas)
        # Calculate the angle between the principal axes and original cartesian reference frame
        phi = np.arctan2(e_vecs[1, 0], e_vecs[0, 0])
        rot_matrix = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        # The sqrt of the eigenvalues are the lengths of the major and minor axes
        transformation_matrix = np.diag(np.sqrt(e_vals)) 
        # Rotate principle strains into original cartesian reference frame
        transformation_matrix = rot_matrix @ transformation_matrix @ rot_matrix.T

        # strain with respect to reference ellipse
        transformation_matrix =  transformation_matrix @ np.linalg.inv(transformation_matrix_ref)

        # Get measured strain values
        exx_fit = transformation_matrix[0, 0] - 1
        eyy_fit = transformation_matrix[1, 1] - 1
        exy_fit = 0.5*(transformation_matrix[0, 1] + transformation_matrix[1, 0])

        # output strain
        strain_map.get_slice('e_xx').data[rx,ry] = exx_fit
        strain_map.get_slice('e_yy').data[rx,ry] = eyy_fit
        strain_map.get_slice('e_xy').data[rx,ry] = exy_fit
        
    if return_ref:
        return strain_map, ABC_ref    
    else:
        return strain_map
