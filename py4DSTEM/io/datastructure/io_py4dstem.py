# Functions for reading and writing subclasses of the base EMD types

import numpy as np
import h5py

from .io_emd import Array_from_h5, Metadata_from_h5

from ...tqdmnd import tqdmnd




# DataCube

# read

def DataCube_from_h5(group:h5py.Group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read
    mode.  Determines if an Array object of this name exists inside this group,
    and if it does, loads and returns it as a DataCube. If it doesn't exist, or if
    it exists but does not have a rank of 4, raises an exception.

    Accepts:
        group (HDF5 group)

    Returns:
        A DataCube instance
    """
    datacube = Array_from_h5(group)
    datacube = DataCube_from_Array(datacube)
    return datacube

def DataCube_from_Array(array):
    """
    Converts an Array to a DataCube.

    Accepts:
        array (Array)

    Returns:
        datacube (DataCube)
    """
    from .datacube import DataCube
    assert(array.rank == 4), "Array must have 4 dimensions"
    array.__class__ = DataCube
    array.__init__(
        data = array.data,
        name = array.name,
        R_pixel_size = [array.dims[0][1]-array.dims[0][0],
                        array.dims[1][1]-array.dims[1][0]],
        R_pixel_units = [array.dim_units[0],
                         array.dim_units[1]],
        Q_pixel_size = [array.dims[2][1]-array.dims[2][0],
                        array.dims[3][1]-array.dims[3][0]],
        Q_pixel_units = [array.dim_units[2],
                         array.dim_units[3]],
        slicelabels = array.slicelabels
    )
    return array





# DiffractionSlice

# read

def DiffractionSlice_from_h5(group:h5py.Group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in
    read mode. Determines if it's a valid Array, and if so loads and
    returns it as a DiffractionSlice. Otherwise, raises an exception.

    Accepts:
        group (HDF5 group)

    Returns:
        A DiffractionSlice instance
    """
    diffractionslice = Array_from_h5(group)
    diffractionslice = DiffractionSlice_from_Array(diffractionslice)
    return diffractionslice


def DiffractionSlice_from_Array(array):
    """
    Converts an Array to a DiffractionSlice.

    Accepts:
        array (Array)

    Returns:
        (DiffractionSlice)
    """
    from .diffractionslice import DiffractionSlice
    assert(array.rank == 2), "Array must have 2 dimensions"
    array.__class__ = DiffractionSlice
    array.__init__(
        data = array.data,
        name = array.name,
        pixel_size = [array.dims[0][1]-array.dims[0][0],
                      array.dims[1][1]-array.dims[1][0]],
        pixel_units = [array.dim_units[0],
                       array.dim_units[1]],
        slicelabels = array.slicelabels
    )
    return array




# RealSlice

# Reading

def RealSlice_from_h5(group:h5py.Group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in
    read mode. Determines if it's a valid Array, and if so loads and
    returns it as a RealSlice. Otherwise, raises an exception.

    Accepts:
        group (HDF5 group)

    Returns:
        A RealSlice instance
    """
    realslice = Array_from_h5(group)
    realslice = RealSlice_from_Array(realslice)
    return realslice


def RealSlice_from_Array(array):
    """
    Converts an Array to a RealSlice.

    Accepts:
        array (Array)

    Returns:
        (RealSlice)
    """
    from .realslice import RealSlice
    assert(array.rank == 2), "Array must have 2 dimensions"
    array.__class__ = RealSlice
    array.__init__(
        data = array.data,
        name = array.name,
        pixel_size = [array.dims[0][1]-array.dims[0][0],
                      array.dims[1][1]-array.dims[1][0]],
        pixel_units = [array.dim_units[0],
                       array.dim_units[1]],
        slicelabels = array.slicelabels
    )
    return array




# Calibration

# read
def Calibration_from_h5(group:h5py.Group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in
    read mode. Determines if it's a valid Metadata representation, and
    if so loads and returns it as a Calibration instance. Otherwise,
    raises an exception.

    Accepts:
        group (HDF5 group)

    Returns:
        A Calibration instance
    """
    cal = Metadata_from_h5(group)
    cal = Calibration_from_Metadata(cal)
    return cal

def Calibration_from_Metadata(metadata):
    """
    Converts a Metadata instance to a Calibration instance.

    Accepts:
        metadata (Metadata)

    Returns:
        (Calibration)
    """
    from .calibration import Calibration
    p = metadata._params
    metadata.__class__ = Calibration
    metadata.__init__(
        name = metadata.name
    )
    metadata._params.update(p)

    return metadata




