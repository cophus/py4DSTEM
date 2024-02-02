# Functions for drift correction


import numpy as np
import matplotlib.pyplot as plt
# import warnings

# from py4DSTEM.io import Datacube
from scipy.interpolate import griddata, interpn


class DriftCorr:
    """
    A class storing DriftCorr data and functions.

    """

    def __init__(
        self,
        image_list = None,
        image_stack = None,
        padding = (100,100),
    ):
        """
        Initialize DriftCorr class with inputs and variables.

        Parameters
        ----------
        image_stack: np.array
            Image data used for the drift correction, with size [num_images num_rows num_cols]
        image_list: list
            Image data used for the drift correction, stored in a list as [im0, im1, ..., imN] for N+1 images
        reshaping_method: str, optional
        
        Returns
        --------
        self: DriftCorr

        """

        if image_list is not None:
            self.image_list = image_list
        elif image_stack is not None:
            self.image_list = temp = [image_stack[ind] for  ind in range(image_stack.shape[0])]
            # self.image_stack = image_stack
            # self.image_stack = np.concatenate(
            #     [
            #         dc0.tree('dark_field').data[],
            #         dc1.tree('dark_field').data,
            #     ],
            #     axis = 0,
            # )
            pass
        else:
            raise Exception('Drift correction requires you to provide input images')

