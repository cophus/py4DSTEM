import matplotlib.pyplot as plt
import numpy as np

from py4DSTEM.datacube import DataCube
from py4DSTEM.process.diffraction import Crystal
from py4DSTEM.process.phase.utils import copy_to_device

from scipy.spatial.transform import Rotation as R

from typing import Sequence

try:
    import cupy as cp

    get_array_module = cp.get_array_module
except (ImportError, ModuleNotFoundError):
    cp = None

    def get_array_module(*args):
        return np


class Tomography:
    """ """

    def __init__(
        self,
        datacubes: Sequence[DataCube] = None,
        rotation: Sequence[np.ndarray] = None,
        translaton: Sequence[np.ndarray] = None,
        initial_object_guess: np.ndarray = None,
        verbose: bool = True,
        device: str = "cpu",
        storage: str = "cpu",
        clear_fft_cache: bool = True,
        name: str = "tomography",
    ):
        """ """

        self._datacubes = datacubes
        self._rotation = rotation
        self._translaton = translaton
        self._verbose = verbose

        self.set_device(device, clear_fft_cache)
        self.set_storage(storage)

    def forward(
        self,
        current_object: np.ndarray,
        tilt_deg: int,
        x_index: int,
        num_points: np.ndarray = 60,
    ):
        """
        Forward projection of object

        Parameters
        ----------
        current_object: np.ndarray
            current object estimate
        tilt_deg: float
            tilt of object in degrees
        x_index: int
            x slice of object to be sliced
        num_points: float
            number of points for bilinear interpolation

        Returns
        --------
        current_object_sliced: np.ndarray
            projection of current object sliced in diffraciton space
        """
        current_object_projected = self.real_space_radon(
            current_object,
            tilt_deg,
            x_index,
            num_points,
        )

        current_object_sliced = self.diffraction_space_slice(
            current_object_projected,
            tilt_deg,
        )

        return current_object_sliced

    def real_space_radon(
        self,
        current_object: np.ndarray,
        tilt_deg: int,
        x_index: int,
        num_points: int,
    ):
        """
        Real space projection of current object

        Parameters
        ----------
        current_object: np.ndarray
            current object estimate
        tilt_deg: float
            tilt of object in degrees
        x_index: int
            x slice of object to be sliced
        num_points: float
            number of points for bilinear interpolation

        Returns
        --------
        current_object_projected: np.ndarray
            projection of current object

        """
        xp = self._xp

        s = current_object.shape

        tilt = xp.deg2rad(tilt_deg)

        padding = int(xp.ceil(xp.abs(xp.tan(tilt) * s[2])))

        line_z = xp.arange(0, 1, 1 / num_points) * (s[2] - 1)
        line_y = line_z * xp.tan(tilt) + padding

        offset = xp.arange(s[1], dtype="int")

        current_object_reshape = xp.pad(
            xp.asarray(current_object[x_index]),
            ((padding, padding), (0, 0), (0, 0), (0, 0), (0, 0)),
        ).reshape(((s[1] + padding * 2) * s[2], s[3], s[4], s[5]))

        current_object_projected = xp.zeros((s[1], s[3], s[4], s[5]))

        yF = xp.floor(line_y).astype("int")
        zF = xp.floor(line_z).astype("int")
        dy = line_y - yF
        dz = line_z - zF

        for basis_index in range(4):
            match basis_index:
                case 0:
                    inds = [yF, zF]
                    weights = (1 - dy) * (1 - dz)
                case 1:
                    inds = [yF + 1, zF]
                    weights = (dy) * (1 - dz)
                case 2:
                    inds = [yF, zF + 1]
                    weights = (1 - dy) * (dz)
                case 3:
                    inds = [yF + 1, zF + 1]
                    weights = (dy) * (dz)

            indy = xp.tile(inds[0], (s[1], 1)) + offset[:, None]
            indz = xp.tile(inds[1], (s[1], 1))
            current_object_projected += (
                current_object_reshape[
                    xp.ravel_multi_index(
                        (indy, indz), (s[1] + 2 * padding, s[2]), mode="clip"
                    )
                ]
                * xp.tile(weights, (s[1], 1))[:, :, None, None, None]
            ).sum(1)

        current_object_projected

        return current_object_projected

    def diffraction_space_slice(
        self,
        current_object_projected: np.ndarray,
        tilt_deg: int,
    ):
        """
        Slicing of diffraction space for rotated object

        Parameters
        ----------
        current_object_rotated: np.ndarray
            current object estimate projected
        tilt_deg: float
            tilt of object in degrees

        Returns
        --------
        current_object_sliced: np.ndarray
            projection of current object sliced in diffraciton space

        """
        xp = self._xp

        s = current_object_projected.shape

        tilt = xp.deg2rad(tilt_deg)

        l = s[-1] * xp.cos(tilt)
        line_y_diff = xp.arange(-1 * (l) / 2, l / 2, l / s[-1])
        line_z_diff = line_y_diff * xp.tan(tilt)

        line_y_diff[line_y_diff < 0] = s[-1] + line_y_diff[line_y_diff < 0]
        line_z_diff[line_y_diff < 0] = s[-1] + line_z_diff[line_y_diff < 0]

        order = xp.argsort(line_y_diff)
        line_y_diff = line_y_diff[order]
        line_z_diff = line_z_diff[order]

        yF_diff = xp.floor(line_y_diff).astype("int")
        zF_diff = xp.floor(line_z_diff).astype("int")
        dy_diff = line_y_diff - yF_diff
        dz_diff = line_z_diff - zF_diff

        current_object_sliced = xp.zeros((s[0], s[-1], s[-1]))
        current_object_projected = xp.pad(
            current_object_projected, ((0, 0), (0, 0), (0, 1), (0, 1))
        )

        for basis_index in range(4):
            match basis_index:
                case 0:
                    inds = [yF_diff, zF_diff]
                    weights = (1 - dy_diff) * (1 - dz_diff)
                case 1:
                    inds = [yF_diff + 1, zF_diff]
                    weights = (dy_diff) * (1 - dz_diff)
                case 2:
                    inds = [yF_diff, zF_diff + 1]
                    weights = (1 - dy_diff) * (dz_diff)
                case 3:
                    inds = [yF_diff + 1, zF_diff + 1]
                    weights = (dy_diff) * (dz_diff)

            current_object_sliced += (
                current_object_projected[:, :, inds[0], inds[1]]
                * weights[None, None, :]
            )

        current_object_sliced

        return self._asnumpy(current_object_sliced)

    def make_test_object(
        self,
        sx: int,
        sy: int,
        sz: int,
        sq: int,
        q_max: float,
        r: int,
        num: int,
    ):
        """
        Make test object with 3D gold cubes at random orientations

        Parameters
        ----------
        sx: int
            x size (pixels)
        sy: int
            y size (pixels)
        sz: int
            z size (pixels)
        sq: int
            q size (pixels)
        q_max: float
            maximum scattering angle (A^-1)
        r: int
            length of 3D gold cubes
        num: int
            number of cubes

        Returns
        --------
        test_object: np.ndarray
            6D test object
        """
        xp_storage = self._xp_storage

        test_object = xp_storage.zeros((sx, sy, sz, sq, sq, sq))

        diffraction_cloud = self.make_diffraction_cloud(sq, q_max, [0, 0, 0])

        test_object[:, :, :, 0, 0, 0] = diffraction_cloud.sum()

        for a0 in range(num):
            s1 = xp_storage.random.randint(r, sx - r)
            s2 = xp_storage.random.randint(r, sy - r)
            h = xp_storage.random.randint(r, sz - r, size=1)
            t = xp_storage.random.randint(0, 360, size=3)

            cloud = xp_storage.asarray(self.make_diffraction_cloud(sq, q_max, t))

            test_object[s1 - r : s1 + r, s2 - r : s2 + r, h[0] - r : h[0] + r] = cloud

        return test_object

    def make_diffraction_cloud(
        self,
        sq,
        q_max,
        rot,
    ):
        """
        Make 3D diffraction cloud

        Parameters
        ----------
        sq: int
            q size (pixels)
        q_max: float
            maximum scattering angle (A^-1)
        rot: 3-tuple
            rotation of cloud

        Returns
        --------
        diffraction_cloud: np.ndarray
            3D structure factor

        """
        xp = self._xp

        gold = self.make_gold(q_max)

        diffraction_cloud = xp.zeros((sq, sq, sq))

        q_step = q_max * 2 / (sq - 1)

        qz = xp.fft.ifftshift(xp.arange(sq) * q_step - q_step * (sq - 1) / 2)
        qx = xp.fft.ifftshift(xp.arange(sq) * q_step - q_step * (sq - 1) / 2)
        qy = xp.fft.ifftshift(xp.arange(sq) * q_step - q_step * (sq - 1) / 2)

        qxa, qya, qza = xp.meshgrid(qx, qy, qz, indexing="ij")

        g_vecs = gold.g_vec_all.copy()
        r = R.from_euler("zxz", [rot[0], rot[1], rot[2]])
        g_vecs = r.as_matrix() @ g_vecs

        cut_off = 0.1

        for a0 in range(gold.g_vec_all.shape[1]):
            bragg_spot = g_vecs[:, a0]
            distance = xp.sqrt(
                (qxa - bragg_spot[0]) ** 2
                + (qya - bragg_spot[1]) ** 2
                + (qza - bragg_spot[2]) ** 2
            )

            update_index = distance < cut_off
            update = xp.zeros((distance.shape))
            update[update_index] = cut_off - distance[update_index]
            update -= xp.min(update)
            update /= xp.sum(update)
            update *= gold.struct_factors_int[a0]
            diffraction_cloud += update

        return diffraction_cloud

    def make_gold(
        self,
        q_max,
    ):
        """
        Calculate structure factor for gold up to q_max

        Parameters
        ----------
        q_max: float
            maximum scattering angle (A^-1)

        Returns
        --------
        crystal: Crystal
            gold crystal with structure factor calculated to q_max

        """

        pos = [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
        atom_num = 79
        a = 4.08
        cell = a

        crystal = Crystal(pos, atom_num, cell)

        crystal.calculate_structure_factors(q_max)

        return crystal

    def set_device(self, device, clear_fft_cache):
        """
        Sets calculation device.

        Parameters
        ----------
        device: str
            Calculation device will be perfomed on. Must be 'cpu' or 'gpu'

        Returns
        --------
        self: PhaseReconstruction
            Self to enable chaining
        """

        if clear_fft_cache is not None:
            self._clear_fft_cache = clear_fft_cache

        if device is None:
            return self

        if device == "cpu":
            import scipy

            self._xp = np
            self._scipy = scipy

        elif device == "gpu":
            from cupyx import scipy

            self._xp = cp
            self._scipy = scipy

        else:
            raise ValueError(f"device must be either 'cpu' or 'gpu', not {device}")

        self._device = device

        return self

    def set_storage(self, storage):
        """
        Sets storage device.

        Parameters
        ----------
        storage: str
            Device arrays will be stored on. Must be 'cpu' or 'gpu'

        Returns
        --------
        self: PhaseReconstruction
            Self to enable chaining
        """

        if storage == "cpu":
            self._xp_storage = np

        elif storage == "gpu":
            if self._xp is np:
                raise ValueError("storage='gpu' and device='cpu' is not supported")
            self._xp_storage = cp

        else:
            raise ValueError(f"storage must be either 'cpu' or 'gpu', not {storage}")

        self._asnumpy = copy_to_device
        self._storage = storage

        return self
