# Functions for drift correction


import numpy as np
import matplotlib.pyplot as plt
# import warnings

from scipy.special import comb
from scipy.ndimage import gaussian_filter
# from py4DSTEM.io import Datacube
# from scipy.interpolate import griddata, interpn
# from scipy.interpolate import interpn
# import scipy.interpolate as spint
# import scipy.spatial.qhull as qhull

class DriftCorr:
    """
    A class storing DriftCorr data and functions.

    """

    def __init__(
        self,
        scan_dir_degrees,
        image_list = None,
        image_stack = None,
        padding = (16,16),
        basis_size = (4,1),
    ):
        """
        Initialize DriftCorr class with inputs and variables.

        Parameters
        ----------
        scan_dir_degrees: np.array, list, tuple
            Scan directions given in degrees.
        image_stack: np.array
            Image data used for the drift correction, with size [num_images num_rows num_cols]
        image_list: list
            Image data used for the drift correction, stored in a list as [im0, im1, ..., imN] for N+1 images
        padding: (int,int)
            Number of pixels to padd with on each side

        
        Returns
        --------
        self: DriftCorr

        """

        # input variables
        self.scan_dir_degrees = np.array(scan_dir_degrees)
        self.scan_dir_rad = np.deg2rad(self.scan_dir_degrees)
        self.padding = np.array(padding)
        self.basis_size = np.array(basis_size)

        # images
        if image_list is not None:
            self.image_list = image_list.copy()
        elif image_stack is not None:
            self.image_list = [image_stack[ind] for ind in range(image_stack.shape[0])]
            # self.image_list = []
            # for a0 in range(image_stack.shape[0]):
            #     self.image_list.append(image_stack[a0])
        else:
            raise Exception('Drift correction requires you to provide input images')

        # size of reconstruction image
        self.output_shape = np.array((0,0), dtype='int')
        self.num_images = len(self.image_list)        
        for a0 in range(self.num_images):
            ct = np.cos(self.scan_dir_rad[a0])
            st = np.sin(self.scan_dir_rad[a0])
            x = self.image_list[a0].shape[0]*ct - self.image_list[a0].shape[1]*st 
            y = self.image_list[a0].shape[0]*st + self.image_list[a0].shape[1]*ct
            self.output_shape[0] = np.maximum(self.output_shape[0], np.ceil(x))
            self.output_shape[1] = np.maximum(self.output_shape[1], np.ceil(y))
        self.output_shape += padding

        # coordinates for reconstruction
        xim = np.arange(self.output_shape[0])
        yim = np.arange(self.output_shape[1])
        xim_a,yim_a = np.meshgrid(xim, yim, indexing='ij')
        self.output_xy = [xim, yim]
        self.output_points = np.vstack((xim_a.ravel(), yim_a.ravel())).T

        # coordinate system    
        self.basis = []  
        self.coefs = []
        for a0 in range(self.num_images):
            # basis coordinates
            u = np.linspace(0.0,1.0,self.image_list[a0].shape[0])
            v = np.linspace(0.0,1.0,self.image_list[a0].shape[1])
            # ua,va = np.meshgrid(u,v,indexing='ij')
            # ua = ua.ravel()
            # va = va.ravel()
            self.basis.append(np.zeros((
                self.image_list[a0].size,
                np.prod(self.basis_size+1),
            )))
            # all basis elements
            for ax in range(self.basis_size[0]+1):
                wu = comb(self.basis_size[0],ax) * (u**ax) * (1-u)**(self.basis_size[0]-ax)

                for ay in range(self.basis_size[1]+1):
                    wv = comb(self.basis_size[1],ay) * (v**ay) * (1-v)**(self.basis_size[1]-ay)

                    ind = ax * (self.basis_size[1]+1) + ay
                    self.basis[a0][:,ind] = (wu[:,None] * wv[None,:]).ravel()

            # image coordinates - consistent with numpy.rot90()
            x0 = (u - np.mean(u)) * (self.image_list[a0].shape[0]-1)
            y0 = (v - np.mean(v)) * (self.image_list[a0].shape[1]-1)

            ct = np.cos(self.scan_dir_rad[a0])
            st = np.sin(self.scan_dir_rad[a0])
            x = x0*ct - y0*st + self.output_shape[0]/2
            y = x0*st + y0*ct + self.output_shape[1]/2
            xa,ya = np.meshgrid(x,y,indexing='ij')
            coords = np.vstack((xa.ravel(),ya.ravel())).T

            print(x0)
            print(xa)
            print(ya)
            # print(np.round(ya))


            # initial coefficients
            self.coefs.append(np.linalg.lstsq(
                self.basis[a0],
                coords,
                rcond = None,
            )[0])

            # xy = self.basis[a0] @ self.coefs[a0]



        # print(np.round(self.basis[0]))
        # print(np.round(self.coefs[0]))
        # print()
        # print(np.round(self.basis[1]))
        # print(np.round(self.coefs[1]))


        # initial image stack
        self.fill_values = np.zeros(self.num_images)
        self.stack_output = np.zeros((
            self.num_images,
            self.output_shape[0],
            self.output_shape[1],
        ))
        for a0 in range(self.num_images):
            self.fill_values[a0] = np.median(self.image_list[a0])
            self.stack_output[a0] = image_transform(
                self.image_list[a0],
                self.basis[a0],
                self.coefs[a0],
                self.output_shape,
                self.fill_values[a0],
            )

        # initial alignment
        G1 = np.fft.fft2(self.stack_output)




    def get_probe_positions(
        self,
        remove_padding = True,
    ):
        """
        Get the current probe positions

        Parameters
        ----------
        self: DriftCorr
            Drift correction class
        remove_padding: bool
            Shift the probes to the unpadded coordinate system
        
        Returns
        --------
        probe_xy: numpy.array
            Array containing all probe positions

        """

        probe_xy = []
        for a0 in range(self.num_images):
            xy = self.basis[a0] @ self.coefs[a0]
            if remove_padding:
                xy -= self.padding[None,:]
            probe_xy.append(xy)

        return probe_xy



def image_transform(
    im,
    basis,
    coefs,
    output_shape,
    fill_value,
    sigma = 0.5
    ):
    # Generate coordinates
    xy = basis @ coefs

    # bilinear coordinates
    xF = np.floor(xy[:,0]).astype('int')
    yF = np.floor(xy[:,1]).astype('int')
    dx = xy[:,0] - xF
    dy = xy[:,1] - yF

    # bilinear interpolation - image
    im_output = np.reshape(
        np.bincount(
        np.ravel_multi_index((
            np.clip(xF  ,0,output_shape[0]-1),
            np.clip(yF  ,0,output_shape[1]-1),
            ), output_shape),
        weights = im.ravel() * (1-dx) * (1-dy),
        minlength = np.prod(output_shape)) \
        + \
        np.bincount(
        np.ravel_multi_index((
            np.clip(xF+1,0,output_shape[0]-1),
            np.clip(yF  ,0,output_shape[1]-1),
            ), output_shape),
        weights = im.ravel() * (  dx) * (1-dy),
        minlength = np.prod(output_shape)) \
        + \
        np.bincount(
        np.ravel_multi_index((
            np.clip(xF  ,0,output_shape[0]-1),
            np.clip(yF+1,0,output_shape[1]-1),
            ), output_shape),
        weights = im.ravel() * (1-dx) * (  dy),
        minlength = np.prod(output_shape)) \
        + \
        np.bincount(
        np.ravel_multi_index((
            np.clip(xF+1,0,output_shape[0]-1),
            np.clip(yF+1,0,output_shape[1]-1),
            ), output_shape),
        weights = im.ravel() * (  dx) * (  dy),
        minlength = np.prod(output_shape)),
        output_shape)

    # bilinear interpolation - count density
    im_count = np.reshape(
        np.bincount(
        np.ravel_multi_index((
            np.clip(xF  ,0,output_shape[0]-1),
            np.clip(yF  ,0,output_shape[1]-1),
            ), output_shape),
        weights = (1-dx) * (1-dy),
        minlength = np.prod(output_shape)) \
        + \
        np.bincount(
        np.ravel_multi_index((
            np.clip(xF+1,0,output_shape[0]-1),
            np.clip(yF  ,0,output_shape[1]-1),
            ), output_shape),
        weights = (  dx) * (1-dy),
        minlength = np.prod(output_shape)) \
        + \
        np.bincount(
        np.ravel_multi_index((
            np.clip(xF  ,0,output_shape[0]-1),
            np.clip(yF+1,0,output_shape[1]-1),
            ), output_shape),
        weights = (1-dx) * (  dy),
        minlength = np.prod(output_shape)) \
        + \
        np.bincount(
        np.ravel_multi_index((
            np.clip(xF+1,0,output_shape[0]-1),
            np.clip(yF+1,0,output_shape[1]-1),
            ), output_shape),
        weights = (  dx) * (  dy),
        minlength = np.prod(output_shape)),
        output_shape)

    # KDE
    im_output = gaussian_filter(
        im_output,
        sigma,
        mode = 'nearest')
    im_count = gaussian_filter(
        im_count,
        sigma,
        mode = 'nearest')
    sub = im_count > 1e-3
    im_output[sub] /= im_count[sub]
    weight = np.clip(im_count, 0.0, 1.0)
    im_output = weight*im_output + (1-weight)*fill_value


    return im_output



    # print(np.reshape(xy[:,0],im.shape).shape)
    # print(im.shape)
    # print(xy_output[1])

    # return interpn(
    #     xy_output, 
    #     im, 
    #     (
    #         np.reshape(xy[:,0],im.shape), 
    #         np.reshape(xy[:,1],im.shape), 
    #     ), 
    #     # im,
    #     # xy_output,
    #     method='linear',
    #     fill_value=np.mean(im),
    #     bounds_error=False,
    # )



# def interp_weights(xy, uv, d=2):
#     tri = qhull.Delaunay(xy)
#     simplex = tri.find_simplex(uv)
#     vertices = np.take(tri.simplices, simplex, axis=0)
#     temp = np.take(tri.transform, simplex, axis=0)
#     delta = uv - temp[:, d]
#     bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
#     return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

# # def interpolate(values, vtx, wts):
# #     return np.einsum('nj,nj->n', np.take(values, vtx), wts)

# def interpolate(values, vtx, wts, fill_value=np.nan):
#     ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
#     ret[np.any(wts < 0, axis=1)] = fill_value
#     return ret



# # precompute interpolants
# self.output_fill = np.zeros(self.num_images)
# for a0 in range(self.num_images):
#     xy = self.basis[a0] @ self.coefs[a0]


    
#     vtx_all[a0], wts_all[a0] = interp_weights(
#         self.output_points, 
#         xy
#     )    

#     # # Precompute all interpolants
#     # # Adapted from https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
#     # vtx_all = np.zeros((data.shape[0],data.shape[1]*data.shape[2],3), dtype=('int'))
#     # wts_all = np.zeros((data.shape[0],data.shape[1]*data.shape[2],3), dtype=('float32'))
#     # fill_all = np.mean(data,axis=(0,1,2))

#     # self.stack_output[a0] = image_transform(
#     #     self.image_list[a0],
#     #     self.basis[0],
#     #     self.coefs[0],
#     #     self.output_xy,
#     # )

