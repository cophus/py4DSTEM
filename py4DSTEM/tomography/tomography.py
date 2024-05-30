import matplotlib.pyplot as plt
import numpy as np

from py4DSTEM.datacube import DataCube
from py4DSTEM.process.diffraction import Crystal
from py4DSTEM.process.phase.utils import copy_to_device
from py4DSTEM.process.calibration import get_origin, fit_origin
from py4DSTEM.utils import get_shifted_ar
from py4DSTEM.preprocess.utils import bin2D

from scipy.spatial.transform import Rotation as R
from scipy.ndimage import zoom

from typing import Sequence, Union, Tuple

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
        datacubes: Union[Sequence[DataCube], Sequence[str]] = None,
        import_kwargs: dict = {},
        object_shape_x_y_z: Tuple = None,
        voxel_size_A: float = None,
        datacube_R_pixel_size_A: float = None,
        datacube_Q_pixel_size_inv_A: float = None,  # do we even need this?
        tilt_deg: Sequence[np.ndarray] = None,
        translation_px: Sequence[np.ndarray] = None,
        scanning_to_tilt_rotation: float = None,
        initial_object_guess: np.ndarray = None,
        verbose: bool = True,
        device: str = "cpu",
        storage: str = "cpu",
        clear_fft_cache: bool = True,
        name: str = "tomography",
    ):
        """ """

        self._datacubes = datacubes
        self._import_kwargs = import_kwargs
        self._object_shape_x_y_z = object_shape_x_y_z
        self._voxel_size_A = voxel_size_A
        self._datacube_R_pixel_size_A = datacube_R_pixel_size_A
        self._datacube_Q_pixel_size_inv_A = datacube_Q_pixel_size_inv_A
        self._tilt_deg = tilt_deg
        self._translation_px = translation_px
        self._scanning_to_tilt_rotation = scanning_to_tilt_rotation
        self._verbose = verbose
        self._initial_object_guess = initial_object_guess

        self.set_device(device, clear_fft_cache)
        self.set_storage(storage)

    def preprocess(
        self,
        diffraction_intensities_shape: int = None,
        resizing_method: str = "bin",
        bin_real_space: int = None,
        crop_reciprocal_space: float = None,
        q_max_inv_A: int = None,
        force_q_to_r_rotation_deg: float = None,
        force_q_to_r_transpose: bool = None,
        diffraction_space_mask_com=None,
        force_centering_shifts: Sequence[Tuple] = None,
        masks_real_space: Union[np.ndarray, Sequence[np.ndarray]] = None,
        r: float = None,
        rscale: float = 1.2,
        fast_center: bool = False,
        fitfunction: str = "plane",
        robust: bool = False,
        robust_steps: int = 3,
        robust_thresh: int = 2,
    ):
        """
        diffraction_intensites_shape: int
            shape of diffraction patterns to reshape data into
        resizing_method: float
            method to reshape diffraction space ("bin", "fourier", "bilinear")
        bin_real_space: int
            factor for binnning in real space
        crop_reciprocal_space: float
            if not None, crops reciprocal space on all sides by integer
        q_max_inv_A: int
            maximum q in inverse angstroms
        force_q_to_r_rotation_deg:float
            force q to r rotation in degrees. If False solves for rotation
            with datacube specified with `datacube_to_solve_rotation` using
            center of mass method.
        force_q_to_r_transpose: bool
            force q to r transpose. If False, solves for transpose
            with datacube specified with `datacube_to_solve_rotation` using
            center of mass method.
        diffraction_space_mask_com: np.ndarray
            applies mask to datacube while solving for CoM rotation
        force_centering_shifts: list of 2-tuples of np.ndarrays of Rshape
            forces the qx and qy shifts of diffraction patterns
        masks_real_space: list of np.ndarray or np.ndarray
            mask for real space. can be the same for each datacube of individually specified.
        r: (float or None)
            the approximate radius of the center disk. If None (default),
            tries to compute r using the get_probe_size method.  The data used for this
            is controlled by dp_max.
        rscale (float)
             expand 'r' by this amount to form a mask about the center disk
            when taking its center of mass
        fast_center: (bool)
            skip the center of mass refinement step.
            arrays are returned for qx0,qy0
        fitfunction: "str"
            fit function for origin ('plane' or 'parabola' or 'bezier_two' or 'constant').
        robust: bool
            If set to True, origin fit will be repeated with outliers
            removed.
        robust_steps: int
            number of robust iterations performed after initial fit.
        robust_thresh: int
            threshold for including points, in units of root-mean-square (standard deviations) error
            of the predicted values after fitting.
        """
        xp_storage = self._xp_storage

        self._num_datacubes = len(self._datacubes)

        self._diffraction_patterns_projected = []
        self._positions_ang = []
        self._positions_vox = []
        self._positions_vox_F = []
        self._positions_vox_dF = []

        # preprocessing of diffraction data
        for a0 in range(self._num_datacubes):
            # load and preprocess datacube
            (datacube, mask_real_space, diffraction_space_mask_com, q_max_inv_A) = (
                self._prepare_datacube(
                    datacube_number=a0,
                    diffraction_intensities_shape=diffraction_intensities_shape,
                    diffraction_space_mask_com=diffraction_space_mask_com,
                    resizing_method=resizing_method,
                    bin_real_space=bin_real_space,
                    masks_real_space=masks_real_space,
                    crop_reciprocal_space=crop_reciprocal_space,
                    q_max_inv_A=q_max_inv_A,
                )
            )

            # initialize object
            if a0 == 0:
                if self._initial_object_guess:
                    self._object = xp_storage.asarray(self._initial_object_guess)
                else:
                    diffraction_shape = self._initial_datacube_shape[-1]
                    self._object = xp_storage.zeros(
                        self._object_shape_x_y_z
                        + (
                            diffraction_shape,
                            diffraction_shape,
                            diffraction_shape,
                        ),
                    )

            # ellpitical fitting?!

            # hmmm how to handle this? we might need to rotate diffraction patterns
            # solve for QR rotation if necessary
            # if a0 is 0 only
            # if force_transpose is not None and force_com_rotation is not None:
            #     dc = self._datacubes[datacube_to_solve_rotation]
            #     _solve_for_center_of_mass_relative_rotation():

            # initialize positions
            mask_real_space = self._calculate_scan_positions(
                datacube_number=a0,
                mask_real_space=mask_real_space,
            )

            # align and reshape
            if force_centering_shifts:
                qx0_fit = force_centering_shifts[datacube_number][0]
                qy0_fit = force_centering_shifts[datacube_number][1]
            else:
                (qx0_fit, qy0_fit) = self._solve_for_diffraction_pattern_centering(
                    datacube=datacube,
                    r=r,
                    rscale=rscale,
                    fast_center=fast_center,
                    mask_real_space=mask_real_space,
                    fitfunction=fitfunction,
                    robust=robust,
                    robust_steps=robust_steps,
                    robust_thresh=robust_thresh,
                )

            self._reshape_diffraction_patterns(
                datacube_number=a0,
                datacube=datacube,
                mask_real_space=mask_real_space,
                qx0_fit=qx0_fit,
                qy0_fit=qy0_fit,
                q_max_inv_A=q_max_inv_A,
            )

        return self

    def reconstruct(
        self,
        num_iter: int = 1,
        step_size: float = 0.5,
        num_points: int = 60,
    ):
        """ """
        for a0 in range(num_iter):
            for a1 in range(self._num_datacubes):
                (
                    current_object_sliced,
                    diffraction_patterns_weighted,
                ) = self._forward(
                    datacube_number=datacube_number,
                    tilt_deg=self._tilt_deg[a1],
                    num_points=num_points,
                )

                self._adjoint(
                    datacube_number=datacube_number,
                    current_object_sliced=current_object_sliced,
                    diffraction_patterns_weighted=diffraction_patterns_weighted,
                    step_size=step_size,
                    current_object_projected=current_object_projected,
                )

        return self

    def _forward(
        self,
        datacube_number: float,
        tilt_deg: int,
        num_points: int,
    ):
        """
        Forward projection of object for simulation of diffraction data

        Parameters
        ----------
        datacube_number: int
            index of datacube
        tilt_deg: float
            tilt of object in degrees
        num_points: float
            number of points for bilinear interpolation

        Returns
        --------
        current_object_sliced: np.ndarray
            projection of current object sliced in diffraciton space
        diffraction_patterns_reshaped: np.ndarray
            datacube with diffraction data reshapped in 2D arrays
        """
        xp = self._xp
        s = self._object.shape
        current_object = xp.asarray(self._object)
        current_object_sliced = xp.zeros((s[0], s[1], s[-1], s[-1]))

        for a0 in range(s[0]):
            current_object_projected = self.real_space_radon(
                current_object=current_object,
                tilt_deg=tilt_deg,
                x_index=a0,
                num_points=num_points,
            )

            current_object_sliced[a0] = self.diffraction_space_slice(
                current_object_projected=current_object_projected,
                tilt_deg=tilt_deg,
            )

        diffraction_patterns_reshaped = self._forward_diffraction_intensities_reshape(
            datacube_number=datacube_number,
        )

        diffraction_patterns_weighted = self._calculate_diffraction_patterns_weighted(
            datacube_number=datacube_number,
            current_object_sliced=current_object_sliced,
            diffraction_patterns_reshaped=diffraction_patterns_reshaped,
        )

        return (
            current_object_sliced,
            diffraction_patterns_weighted,
        )

    # def _adjoint(
    #     self,
    #     datacube_number,
    #     current_object_sliced,
    #     diffraction_patterns_weighted,
    #     step_size,
    #     tilt_deg,
    # ):
    #     """ """
    #     # update sliced object
    #     current_object_sliced = step_size * (
    #         diffraction_patterns_weighted - current_object_sliced
    #     )

    #     for a0 in range(self._object.shape[0]):

    # def diffraction_space_adjoint(
    #     self,
    #     current_object_sliced: np.ndarray,
    #     tilt_deg: int,
    # ):
    #     """
    #     Slicing of diffraction space for rotated object

    #     Parameters
    #     ----------
    #     current_object_rotated: np.ndarray
    #         current object estimate projected
    #     tilt_deg: float
    #         tilt of object in degrees

    #     Returns
    #     --------
    #     current_object_sliced: np.ndarray
    #         projection of current object sliced in diffraciton space

    #     """
    #     xp = self._xp

    #     # s = current_object_projected.shape

    #     tilt = xp.deg2rad(tilt_deg)

    #     line_y_diff = self._line_y_diff
    #     line_z_diff = self._line_z_diff
    #     yF_diff = self._yF_diff
    #     zF_diff = self._zF_diff
    #     dy_diff = self._dy_diff
    #     dz_diff = self._dz_diff

    #     current_object_projected_updated = current_object_projected.copy()

    #     for basis_index in range(4):
    #         match basis_index:
    #             case 0:
    #                 inds = [yF_diff, zF_diff]
    #                 weights = (1 - dy_diff) * (1 - dz_diff)
    #             case 1:
    #                 inds = [yF_diff + 1, zF_diff]
    #                 weights = (dy_diff) * (1 - dz_diff)
    #             case 2:
    #                 inds = [yF_diff, zF_diff + 1]
    #                 weights = (1 - dy_diff) * (dz_diff)
    #             case 3:
    #                 inds = [yF_diff + 1, zF_diff + 1]
    #                 weights = (dy_diff) * (dz_diff)

    #         # current_object_projected -=

    #     return self._asnumpy(current_object_sliced)

    def _prepare_datacube(
        self,
        datacube_number,
        diffraction_intensities_shape,
        diffraction_space_mask_com,
        resizing_method,
        bin_real_space,
        masks_real_space,
        crop_reciprocal_space,
        q_max_inv_A,
    ):
        """
        datacube_number: int
            index of datacube
        diffraction_intensites_shape: int
            shape of diffraction patterns to reshape data into
        diffraction_space_mask_com: np.ndarray
            applies mask to datacube while solving for CoM rotation
        resizing_method: float
            method to reshape diffraction space ("bin", "fourier", "bilinear")
        bin_real_space: int
            factor for binnning in real space
        masks_real_space: list of np.ndarray or np.ndarray
            mask for real space. can be the same for each datacube of individually specified.
        crop_reciprocal_space: float
            if not None, crops reciprocal space on all sides by integer
        q_max_inv_A: int
            maximum q in inverse angtroms
        """
        if type(self._datacubes[datacube_number]) is str:
            try:
                from py4DSTEM import import_file

                datacube = import_file(
                    self._datacubes[datacube_number], **self._import_kwargs
                )

            except:
                from py4DSTEM import read

                datacube = read(self._datacubes[datacube_number], **self._import_kwargs)
        else:
            datacube = self._datacubes[datacube_number]

        if masks_real_space is not None:
            if type(masks_real_space) is np.ndarray:
                mask_real_space = masks_real_space
            else:
                mask_real_space = masks_real_space[datacube_number]
            mask_real_space = np.ndarray(masks_real_space, dtype="bool")
        else:
            mask_real_space = None

        if crop_reciprocal_space is not None:
            datacube.crop_Q(
                (
                    crop_reciprocal_space,
                    -crop_reciprocal_space,
                    crop_reciprocal_space,
                    -crop_reciprocal_space,
                )
            )

        # resize diffraction space
        if diffraction_intensities_shape is not None:
            Q = datacube.shape[-1]
            S = diffraction_intensities_shape
            resampling_factor = S / Q

            if resizing_method == "bin":
                datacube = datacube.bin_Q(N=int(1 / resampling_factor))
            if diffraction_space_mask_com is not None:
                diffraction_space_mask_com = bin2D(
                    diffraction_space_mask_com, int(1 / resampling_factor)
                )

            elif resizing_method == "fourier":
                datacube = datacube.resample_Q(
                    N=resampling_factor, method=resizing_method
                )
                if diffraction_space_mask_com is not None:
                    diffraction_space_mask_com = fourier_resample(
                        diffraction_space_mask_com,
                        output_size=(S, S),
                        force_nonnegative=True,
                    )

            elif resizing_method == "bilinear":
                datacube = datacube.resample_Q(
                    N=resampling_factor, method=resizing_method
                )
                if diffraction_space_mask_com is not None:
                    diffraction_space_mask_com = zoom(
                        diffraction_space_mask_com,
                        (resampling_factor, resampling_factor),
                        order=1,
                    )

            else:
                raise ValueError(
                    (
                        "reshaping_method needs to be one of 'bilinear', 'fourier', or 'bin', "
                        f"not {reshaping_method}."
                    )
                )

            if datacube_number == 0:
                self._datacube_Q_pixel_size_inv_A /= resampling_factor
                if q_max_inv_A is not None:
                    q_max_inv_A *= resampling_factor
                else:
                    q_max_inv_A = (
                        self._datacube_Q_pixel_size_inv_A * datacube.Qshape[0] / 2
                    )
        else:
            if datacube_number == 0 and q_max_inv_A is None:
                q_max_inv_A = self._datacube_Q_pixel_size_inv_A * datacube.Qshape[0] / 2

        # bin real space
        if bin_real_space is not None:
            datacube.bin_R(bin_real_space)
            if mask_real_space is not None:
                mask_real_space = bin2D(mask_real_space, bin_real_space)
                mask_real_space = np.floor(
                    mask_real_space / bin_real_space / bin_real_space
                )
                mask_real_space = np.ndarray(masks_real_space, dtype="bool")
            if datacube_number == 0:
                self._datacube_R_pixel_size_A *= bin_real_space

        self._initial_datacube_shape = datacube.data.shape

        return datacube, mask_real_space, diffraction_space_mask_com, q_max_inv_A

    def _calculate_scan_positions(
        self,
        datacube_number,
        mask_real_space,
    ):
        """
        Calculate scan positions in angstroms and voxels

        Parameters
        ----------
        datacube_number: int
            index of datacube
        mask_real_space: np.ndarray
            mask for real space

        Returns
        --------
        mask_real_space: np.ndarray
            mask for real space

        """
        xp_storage = self._xp_storage

        # calculate shape
        field_of_view_px = self._object.shape[0:2]
        self._field_of_view_A = (
            self._voxel_size_A * field_of_view_px[0],
            self._voxel_size_A * field_of_view_px[1],
        )

        # calculate positions
        s = self._initial_datacube_shape

        step_size = self._datacube_R_pixel_size_A

        x = np.arange(s[0])
        y = np.arange(s[1])

        if self._translation_px is not None:
            x += self._translation_px[datacube_number][0]
            y += self._translation_px[datacube_number][1]

        x *= step_size
        y *= step_size

        x, y = np.meshgrid(x, y, indexing="ij")

        if self._scanning_to_tilt_rotation is not None:
            rotation_angle = np.deg2rad(self._scanning_to_tilt_rotation)
            x, y = x * np.cos(rotation_angle) + y * np.sin(rotation_angle), -x * np.sin(
                rotation_angle
            ) + y * np.cos(rotation_angle)

        # remove data outside FOV
        if mask_real_space is None:
            mask_real_space = np.ones(x.shape, dtype="bool")
        mask_real_space[x > self._field_of_view_A[0]] = False
        mask_real_space[x < 0] = False
        mask_real_space[y > self._field_of_view_A[1]] = False
        mask_real_space[y < 0] = False

        # calculate positions in voxels
        x = x[mask_real_space].ravel()
        y = y[mask_real_space].ravel()

        x_vox = x / self._voxel_size_A
        y_vox = y / self._voxel_size_A

        x_vox_F = np.floor(x_vox).astype("int")
        y_vox_F = np.floor(y_vox).astype("int")
        dx = x_vox - x_vox_F
        dy = y_vox - y_vox_F

        # store pixels
        x = xp_storage.asarray(x)
        y = xp_storage.asarray(y)

        self._positions_ang.append((x, y))
        self._positions_vox.append((x_vox, y_vox))
        self._positions_vox_F.append((x_vox_F, y_vox_F))
        self._positions_vox_dF.append((dx, dy))

        return mask_real_space

    # def _solve_for_center_of_mass_relative_rotation():

    def _solve_for_diffraction_pattern_centering(
        self,
        datacube,
        r,
        rscale,
        fast_center,
        mask_real_space,
        fitfunction,
        robust,
        robust_steps,
        robust_thresh,
    ):
        """
        Solve for qx and qy shifts

        Parameters
        ----------
        r: (float or None)
            the approximate radius of the center disk. If None (default),
            tries to compute r using the get_probe_size method.  The data used for this
            is controlled by dp_max.
        rscale (float)
             expand 'r' by this amount to form a mask about the center disk
            when taking its center of mass
        fast_center: (bool)
            skip the center of mass refinement step.
        mask_real_space: np.ndarray or None
            if not None, should be an (R_Nx,R_Ny) shaped
            boolean array. Origin is found only where mask==True, and masked
            arrays are returned for qx0,qy0
        fitfunction: "str"
            fit function for origin ('plane' or 'parabola' or 'bezier_two' or 'constant').
        robust: bool
            If set to True, origin fit will be repeated with outliers
            removed.
        robust_steps: int
            number of robust iterations performed after initial fit.
        robust_thresh: int
            threshold for including points, in units of root-mean-square (standard deviations) error
            of the predicted values after fitting.

        Returns
        --------
        qx0_fit, qy0_fit: (np.ndarray, np.ndarray)
            qx and qy shifts

        """

        (qx0, qy0, _) = get_origin(
            datacube,
            r=r,
            rscale=rscale,
            mask=mask_real_space,
            fast_center=fast_center,
            verbose=False,
        )

        (qx0_fit, qy0_fit, qx0_res, qy0_res) = fit_origin(
            (qx0, qy0),
            mask=mask_real_space,
            fitfunction=fitfunction,
            returnfitp=False,
            robust=robust,
            robust_steps=robust_steps,
            robust_thresh=robust_thresh,
        )

        return qx0_fit, qy0_fit

    def _reshape_diffraction_patterns(
        self,
        datacube_number,
        datacube,
        mask_real_space,
        qx0_fit,
        qy0_fit,
        q_max_inv_A,
    ):
        """
        Reshapes diffraction data into a 2 column array

        Parameters
        ----------
        datacube_number: int
            index of datacube
        datacube: DataCube
            datacube to be reshapped
        mask_real_space: np.ndarray
            mask for real space
        qx0_fit: np.ndarray
            qx shifts
        qy0_fit: int
            qy shifts
        q_max_inv_A: int
            maximum q in inverse angstroms
        """
        xp_storage = self._xp_storage

        s = self._initial_datacube_shape

        # calculate bincount array
        if datacube_number == 0:
            mask = np.ones((s[-1], s[-1]), dtype="bool")
            mask[:, int(np.ceil(s[-1] / 2)) :] = 0
            mask[: int(np.ceil(s[-1] / 2)), int(np.floor(s[-1] / 2))] = 0

            ind_diffraction = np.roll(
                np.arange(s[-1] * s[-1]).reshape(s[-1], s[-1]),
                (int(np.floor(s[-1] / 2)), int(np.floor(s[-1] / 2))),
                axis=(0, 1),
            )

            ind_diffraction[mask] = 1e10

            a = np.argsort(ind_diffraction.flatten())
            i = np.empty_like(a)
            i[a] = np.arange(a.size)
            i = i.reshape((s[-1], s[-1]))

            ind_diffraction = i
            ind_diffraction_rot = np.rot90(ind_diffraction, 2)

            ind_diffraction[mask] = ind_diffraction_rot[mask]

            self._ind_diffraction = ind_diffraction
            self._ind_diffraction_ravel = ind_diffraction.ravel()
            self._q_length = np.unique(self._ind_diffraction).shape[0] + 1

            # pixels to remove
            q_max_px = q_max_inv_A / self._datacube_Q_pixel_size_inv_A

            x = np.arange(s[-1]) - ((s[-1] - 1) / 2)
            y = np.arange(s[-1]) - ((s[-1] - 1) / 2)
            xx, yy = np.meshgrid(x, y)
            circular_mask = ((xx) ** 2 + (yy) ** 2) ** 0.5 < q_max_px

            self._circular_mask = circular_mask
            self._circular_masK_ravel = circular_mask.ravel()
            self._circular_masK_bincount = np.asarray(
                np.bincount(
                    self._ind_diffraction_ravel,
                    circular_mask.ravel(),
                    minlength=self._q_length,
                ),
                dtype="bool",
            )

        center = (s[-1] / 2, s[-1] / 2)

        diffraction_patterns_reshaped = np.zeros((s[0] * s[1], self._q_length))

        for a0 in range(s[0]):
            for a1 in range(s[0]):
                xF = int(np.floor(qx0_fit[a0, a1]))
                yF = int(np.floor(qy0_fit[a0, a1]))

                wx = qx0_fit[a0, a1] - xF
                wy = qy0_fit[a0, a1] - yF

                dp = datacube.data[a0, a1]

                index = np.ravel_multi_index((a0, a1), (s[0], s[1]))
                diffraction_patterns_reshaped[index] = (
                    (
                        (
                            (1 - wx)
                            * (1 - wy)
                            * np.bincount(
                                self._ind_diffraction_ravel,
                                np.roll(dp, (xF, yF), axis=(0, 1)).ravel(),
                                minlength=self._q_length,
                            )
                        )
                    )
                    + (
                        (wx)
                        * (1 - wy)
                        * np.bincount(
                            self._ind_diffraction_ravel,
                            np.roll(dp, (xF + 1, yF), axis=(0, 1)).ravel(),
                            minlength=self._q_length,
                        )
                    )
                    + (
                        (1 - wx)
                        * (wy)
                        * np.bincount(
                            self._ind_diffraction_ravel,
                            np.roll(dp, (xF, yF + 1), axis=(0, 1)).ravel(),
                            minlength=self._q_length,
                        )
                    )
                    + (
                        (wx)
                        * (wy)
                        * np.bincount(
                            self._ind_diffraction_ravel,
                            np.roll(dp, (xF + 1, yF + 1), axis=(0, 1)).ravel(),
                            minlength=self._q_length,
                        )
                    )
                )

        self._diffraction_patterns_projected.append(
            diffraction_patterns_reshaped[:, self._circular_masK_bincount]
        )

    def _forward_simulation(
        self,
        current_object: np.ndarray,
        tilt_deg: int,
        x_index: int,
        num_points: np.ndarray = 60,
    ):
        """
        Forward projection of object for simulation of diffraction data

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
            current_object[x_index],
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

        return self._asnumpy(current_object_sliced)

    def _forward_diffraction_intensities_reshape(
        self,
        datacube_number,
    ):
        """
        Reshape 1D diffraction data to full patterns

        Parameters
        ----------
        datacube_number: int
            index of datacube

        Returns
        --------
        diffraction_patterns_reshaped: np.ndarray
            datacube with diffraction data reshapped in 2D arrays
        """
        xp = self._xp
        s = self._initial_datacube_shape

        diffraction_patterns_flat = xp.asarray(
            self._diffraction_patterns[datacube_number]
        )
        diffraction_patterns_reshaped = xp.zeros(
            (diffraction_patterns_flat.shape[0], s[-1] * s[-1])
        )

        diffraction_patterns_reshaped[
            :, xp.asarray(self._ind_diffraction_space_ravel)
        ] = diffraction_patterns_flat
        diffraction_patterns_reshaped = diffraction_patterns_reshaped[
            :, xp.asarray(self._reorder_patterns)
        ]

        diffraction_patterns_reshaped = diffraction_patterns_reshaped.reshape(
            (diffraction_patterns_flat.shape[0], s[-1], s[-1])
        )

        diffraction_patterns_reshaped += np.rot90(
            diffraction_patterns_reshaped, 2, axes=(-1, -2)
        )

        return diffraction_patterns_reshaped

    def _calculate_diffraction_patterns_weighted(
        self, datacube_number, current_object_sliced, diffraction_patterns_reshaped
    ):
        """
        Calculate diffraction pattern re-weighted for voxels

        Parameters
        ----------
        datacube_number: int
            index of datacube
        current_object_sliced: np.ndarray
            projection of current object sliced in diffraciton space
        diffraction_patterns_reshaped: np.ndarray
            datacube with diffraction data reshapped in 2D arrays

        Returns
        --------
        diffraction_patterns_weighted: np.ndarray
            diffraction patterns reshaped for update

        """
        xp = self._xp

        xF = xp.asarray(self._positions_vox_F[datacube_number][0])
        yF = xp.asarray(self._positions_vox_F[datacube_number][1])

        dx = xp.asarray(self._positions_vox_dF[datacube_number][0])
        dy = xp.asarray(self._positions_vox_dF[datacube_number][1])

        s = current_object_sliced.shape
        diffraction_patterns_weighted = xp.zeros((s[0] + 1, s[1] + 1, s[2], s[3]))

        for basis_index in range(4):
            match basis_index:
                case 0:
                    inds = [xF, yF]
                    weights = (1 - dx) * (1 - dy)
                case 1:
                    inds = [xF + 1, yF]
                    weights = (dx) * (1 - dy)
                case 2:
                    inds = [xF, yF + 1]
                    weights = (1 - dx) * (dy)
                case 3:
                    inds = [xF + 1, yF + 1]
                    weights = (dx) * (dy)

            diffraction_patterns_weighted[inds[0], inds[1]] += (
                diffraction_patterns_reshaped * weights[:, None, None]
            )

        return diffraction_patterns_weighted[:-1, :-1]

    def _make_test_object(
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

        diffraction_cloud = self._make_diffraction_cloud(sq, q_max, [0, 0, 0])

        test_object[:, :, :, 0, 0, 0] = diffraction_cloud.sum()

        for a0 in range(num):
            s1 = xp_storage.random.randint(r, sx - r)
            s2 = xp_storage.random.randint(r, sy - r)
            h = xp_storage.random.randint(r, sz - r, size=1)
            t = xp_storage.random.randint(0, 360, size=3)

            cloud = xp_storage.asarray(self._make_diffraction_cloud(sq, q_max, t))

            test_object[s1 - r : s1 + r, s2 - r : s2 + r, h[0] - r : h[0] + r] = cloud

        return test_object

    def _make_diffraction_cloud(
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

        gold = self._make_gold(q_max)

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

    def _make_gold(
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
