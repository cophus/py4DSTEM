import numpy as np
from copy import copy
import h5py
from .dataobject import DataObject
from .pointlist import PointList
from ...process.utils import tqdmnd, add_to_2D_array_from_floats


class PointListArray(DataObject):
    """
    An object containing an array of PointLists.
    Facilitates more rapid access of subpointlists which have known, well structured coordinates, such
    as real space scan positions R_Nx,R_Ny.

    Args:
        coordinates: see PointList documentation
        shape (2-tuple of ints): the array shape.  Typically the real space shape
            ``(R_Nx, R_Ny)``.
    """
    def __init__(self, coordinates, shape, dtype=float, **kwargs):
        """
		Instantiate a PointListArray object.
		Creates a PointList with coordinates at each point of a 2D grid with a shape specified
        by the shape argument.
        """
        DataObject.__init__(self, **kwargs)

        self.coordinates = coordinates  #: the coordinates; see the PointList documentation
        self.default_dtype = dtype      #: the default datatype; see the PointList documentation

        assert isinstance(shape,tuple), "Shape must be a tuple."
        assert len(shape) == 2, "Shape must be a length 2 tuple."
        #: the shape of the 2D grid of positions populated with PointList instances;
        #: typically, this will be ``(R_Nx,R_Ny)``
        self.shape = shape

        # Define the data type for the structured arrays in the PointLists
        if isinstance(coordinates, np.dtype):
            self.dtype = coordinates  # the custom datatype; see the PointList documentation
        elif type(coordinates[0])==str:
            self.dtype = np.dtype([(name,self.default_dtype) for name in coordinates])
        elif type(coordinates[0])==tuple:
            self.dtype = np.dtype(coordinates)
        else:
            raise TypeError("coordinates must be a list of strings, or a list of 2-tuples of structure (name, dtype).")

        kwargs['searchable']=False   # Ensure that the subpointlists don't all appear in searches
        self.pointlists = [[PointList(coordinates=self.coordinates,
                            dtype = self.default_dtype,
                            **kwargs) for j in range(self.shape[1])] for i in range(self.shape[0])]

    def get_pointlist(self, i, j):
        """
        Returns the pointlist at i,j
        """
        return self.pointlists[i][j]

    def copy(self, **kwargs):
        """
        Returns a copy of itself.
        """
        new_pointlistarray = PointListArray(coordinates=self.coordinates,
                                            shape=self.shape,
                                            dtype=self.default_dtype,
                                            **kwargs)

        for i in range(new_pointlistarray.shape[0]):
            for j in range(new_pointlistarray.shape[1]):
                curr_pointlist = new_pointlistarray.get_pointlist(i,j)
                curr_pointlist.add_pointlist(self.get_pointlist(i,j).copy())

        return new_pointlistarray

    def add_coordinates(self, new_coords, **kwargs):
        """
        Creates a copy of the PointListArray, but with additional coordinates given by new_coords.
        new_coords must be a string of 2-tuples, ('name', dtype)
        """
        coords = []
        for key in self.dtype.fields.keys():
            coords.append((key,self.dtype.fields[key][0]))
        for coord in new_coords:
            coords.append((coord[0],coord[1]))

        new_pointlistarray = PointListArray(coordinates=coords,
                                            shape=self.shape,
                                            dtype=self.default_dtype,
                                            **kwargs)

        for i in range(new_pointlistarray.shape[0]):
            for j in range(new_pointlistarray.shape[1]):
                curr_pointlist_new = new_pointlistarray.get_pointlist(i,j)
                curr_pointlist_old = self.get_pointlist(i,j)

                data = np.zeros(curr_pointlist_old.length, np.dtype(coords))
                for key in self.dtype.fields.keys():
                    data[key] = np.copy(curr_pointlist_old.data[key])

                curr_pointlist_new.add_dataarray(data)

        return new_pointlistarray



### Read/Write

def save_pointlistarray_group(group, pointlistarray):
    """
    Expects an open .h5 group and a DataCube; saves the DataCube to the group
    """
    try:
        n_coords = len(pointlistarray.dtype.names)
    except:
        n_coords = 1
    #coords = np.string_(str([coord for coord in pointlistarray.dtype.names]))
    group.attrs.create("coordinates", np.string_(str(pointlistarray.dtype)))
    group.attrs.create("dimensions", n_coords)

    pointlist_dtype = h5py.special_dtype(vlen=pointlistarray.dtype)
    name = "data"
    dset = group.create_dataset(name,pointlistarray.shape,pointlist_dtype)

    for (i,j) in tqdmnd(dset.shape[0],dset.shape[1]):
        dset[i,j] = pointlistarray.get_pointlist(i,j).data

def get_pointlistarray_from_grp(g):
    """ Accepts an h5py Group corresponding to a pointlistarray in an open, correctly formatted H5 file,
        and returns a PointListArray.
    """
    name = g.name.split('/')[-1]
    dset = g['data']
    shape = g['data'].shape
    coordinates = g['data'][0,0].dtype
    pla = PointListArray(coordinates=coordinates,shape=shape,name=name)
    for (i,j) in tqdmnd(shape[0],shape[1],desc="Reading PointListArray",unit="PointList"):
        pla.get_pointlist(i,j).add_dataarray(dset[i,j])
    return pla




class PointListSet(DataObject):
    """
    An object containing an array of PointLists.
    Facilitates more rapid access of subpointlists which have known, well structured coordinates, such
    as real space scan positions R_Nx,R_Ny.

    Args:
        field_names: see PointList documentation
        shape (2-tuple of ints): the array shape.  Typically the real space shape
            ``(R_Nx, R_Ny)``.
    """
    def __init__(self, field_names, shape, dtype=float, **kwargs):
        """
        Instantiate a PointListSet object.
        Creates a PointList with field_names at each point of a 2D grid with a shape specified
        by the shape argument.
        """
        DataObject.__init__(self, **kwargs)

        self.field_names = field_names  #: the coordinates; see the PointList documentation
        self.default_dtype = dtype      #: the default datatype; see the PointList documentation

        self.shape = shape
        self.ind_start = np.zeros(shape, dtype='int')
        self.ind_stop = np.zeros(shape, dtype='int')
        self.ind_current = np.array([0], dtype='int')

        self.point_data = PointList(field_names)


    def get_pointlist(self, ax, ay):
        """
        Returns the pointlist at ax,ay
        """
        return self.point_data.data[self.ind_start[ax,ay]:self.ind_stop[ax,ay]]


    def set_pointlist(self, pointlist, ax, ay):
        """
        Stores the pointlist at ax,ay.
        If point list doesn't exist, append to the list.  
        If it does exist, replace it or append it depending on length.
        """

        num_points_init = self.ind_stop[ax,ay] - self.ind_start[ax,ay]
        num_points_add = pointlist.data.shape

        if num_points_init == 0 or num_points_add > num_points_init:
            # print(self.point_data.data.dtype)
            self.point_data.add_pointlist(pointlist)
            self.ind_start[ax,ay] = self.ind_current
            self.ind_stop[ax,ay] = self.ind_current + num_points_add 
            self.ind_current += num_points_add

        elif num_points_add == num_points_init:
            self.point_data[self.ind_start[ax,ay]:self.ind_stop[ax,ay]] = pointlist

        # elif num_points_add < num_points_init:
        else:
            self.ind_stop[ax,ay] = self.ind_start[ax,ay]+num_points_add
            self.point_data[self.ind_start[ax,ay]:self.ind_stop[ax,ay]] = pointlist

    def get_bragg_vector_map(self, Q_Nx, Q_Ny):
        """
        Calculates the Bragg vector map from a PointListSet of Bragg peak positions, given
        braggpeak positions which have been centered about the origin. In the returned array
        braggvectormap, the origin is placed at (Q_Nx/2.,Q_Ny/2.)

        Args:
            braggpeaks (PointListArray): Must have the coords 'qx','qy','intensity', the
                default coordinates from the bragg peak detection fns
            Q_Nx,Q_Ny (ints): the size of diffraction space in pixels

        Returns:
            (ndarray): the bragg vector map
        """

        qx0,qy0 = Q_Nx/2.,Q_Ny/2.
        # for (Rx, Ry) in tqdmnd(self.shape[0],self.shape[1],
        #                        desc='Computing Bragg vector map',unit='DP',unit_scale=True):
        #     peaks = braggpeaks.get_pointlist(Rx,Ry)
        #     for i in range(peaks.length):
        #         qx = peaks.data['qx'][i]+qx0
        #         qy = peaks.data['qy'][i]+qy0
        #         I = peaks.data['intensity'][i]
        #         braggvectormap = add_to_2D_array_from_floats(braggvectormap,qx,qy,I)
        qx = self.point_data.data['qx']+qx0
        qy = self.point_data.data['qy']+qy0
        I = self.point_data.data['intensity']
        # braggvectormap = add_to_2D_array_from_floats(braggvectormap,qx,qy,I)

        # braggvectormap = np.zeros((Q_Nx,Q_Ny))
        # braggvectormap = np.zeros(Q_Nx*Q_Ny)
        inds = np.ravel_multi_index(
            [np.round(qx).astype('int'),np.round(qy).astype('int')],(Q_Nx,Q_Ny)
        )
        braggvectormap = np.bincount(inds, I, minlength=Q_Nx*Q_Ny)
        braggvectormap.shape = (Q_Nx,Q_Ny)

        # numpy.ravel_multi_index(multi_index, dims, mode='raise', order='C')

        return braggvectormap
