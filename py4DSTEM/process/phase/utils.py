from typing import Mapping, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

try:
    import cupy as cp
except ImportError:
    cp = None

from py4DSTEM.process.calibration import fit_origin
from py4DSTEM.process.utils.utils import electron_wavelength_angstrom
from scipy.ndimage import gaussian_filter

#: Symbols for the polar representation of all optical aberrations up to the fifth order.
polar_symbols = (
    "C10",
    "C12",
    "phi12",
    "C21",
    "phi21",
    "C23",
    "phi23",
    "C30",
    "C32",
    "phi32",
    "C34",
    "phi34",
    "C41",
    "phi41",
    "C43",
    "phi43",
    "C45",
    "phi45",
    "C50",
    "C52",
    "phi52",
    "C54",
    "phi54",
    "C56",
    "phi56",
)

#: Aliases for the most commonly used optical aberrations.
polar_aliases = {
    "defocus": "C10",
    "astigmatism": "C12",
    "astigmatism_angle": "phi12",
    "coma": "C21",
    "coma_angle": "phi21",
    "Cs": "C30",
    "C5": "C50",
}


### Probe functions


class ComplexProbe:
    """
    Complex Probe Class.

    Simplified version of CTF and Probe from abTEM:
    https://github.com/abTEM/abTEM/blob/master/abtem/transfer.py
    https://github.com/abTEM/abTEM/blob/master/abtem/waves.py

    Parameters
    ----------
    energy: float
        The electron energy of the wave functions this contrast transfer function will be applied to [eV].
    semiangle_cutoff: float
        The semiangle cutoff describes the sharp Fourier space cutoff due to the objective aperture [mrad].
    gpts : Tuple[int,int]
        Number of grid points describing the wave functions.
    sampling : Tuple[float,float]
        Lateral sampling of wave functions in Å
    device: str, optional
        Device to perform calculations on. Must be either 'cpu' or 'gpu'
    rolloff: float, optional
        Tapers the cutoff edge over the given angular range [mrad].
    focal_spread: float, optional
        The 1/e width of the focal spread due to chromatic aberration and lens current instability [Å].
    angular_spread: float, optional
        The 1/e width of the angular deviations due to source size [mrad].
    gaussian_spread: float, optional
        The 1/e width image deflections due to vibrations and thermal magnetic noise [Å].
    phase_shift : float, optional
        A constant phase shift [radians].
    parameters: dict, optional
        Mapping from aberration symbols to their corresponding values. All aberration magnitudes should be given in Å
        and angles should be given in radians.
    kwargs:
        Provide the aberration coefficients as keyword arguments.
    """

    def __init__(
        self,
        energy: float,
        gpts: Tuple[int, int],
        sampling: Tuple[float, float],
        semiangle_cutoff: float = np.inf,
        rolloff: float = 2.0,
        vacuum_probe_intensity: np.ndarray = None,
        device: str = "cpu",
        focal_spread: float = 0.0,
        angular_spread: float = 0.0,
        gaussian_spread: float = 0.0,
        phase_shift: float = 0.0,
        parameters: Mapping[str, float] = None,
        **kwargs,
    ):

        # Should probably be abstracted away in a device.py similar to:
        # https://github.com/abTEM/abTEM/blob/master/abtem/device.py
        if device == "cpu":
            self._xp = np
            self._asnumpy = np.asarray
        elif device == "gpu":
            self._xp = cp
            self._asnumpy = cp.asnumpy
        else:
            raise ValueError(f"device must be either 'cpu' or 'gpu', not {device}")

        for key in kwargs.keys():
            if (key not in polar_symbols) and (key not in polar_aliases.keys()):
                raise ValueError("{} not a recognized parameter".format(key))

        self._vacuum_probe_intensity = vacuum_probe_intensity
        self._semiangle_cutoff = semiangle_cutoff
        self._rolloff = rolloff
        self._focal_spread = focal_spread
        self._angular_spread = angular_spread
        self._gaussian_spread = gaussian_spread
        self._phase_shift = phase_shift
        self._energy = energy
        self._wavelength = electron_wavelength_angstrom(energy)
        self._gpts = gpts
        self._sampling = sampling

        self._parameters = dict(zip(polar_symbols, [0.0] * len(polar_symbols)))

        if parameters is None:
            parameters = {}

        parameters.update(kwargs)
        self.set_parameters(parameters)

    def set_parameters(self, parameters: dict):
        """
        Set the phase of the phase aberration.
        Parameters
        ----------
        parameters: dict
            Mapping from aberration symbols to their corresponding values.
        """

        for symbol, value in parameters.items():
            if symbol in self._parameters.keys():
                self._parameters[symbol] = value

            elif symbol == "defocus":
                self._parameters[polar_aliases[symbol]] = -value

            elif symbol in polar_aliases.keys():
                self._parameters[polar_aliases[symbol]] = value

            else:
                raise ValueError("{} not a recognized parameter".format(symbol))

        return parameters

    def evaluate_aperture(
        self, alpha: Union[float, np.ndarray], phi: Union[float, np.ndarray] = None
    ) -> Union[float, np.ndarray]:

        xp = self._xp
        semiangle_cutoff = self._semiangle_cutoff / 1000

        if self._vacuum_probe_intensity is not None:
            vacuum_probe_intensity = xp.asarray(
                self._vacuum_probe_intensity, dtype=xp.float32
            )
            vacuum_probe_amplitude = xp.sqrt(xp.maximum(vacuum_probe_intensity, 0))
            return xp.fft.ifftshift(vacuum_probe_amplitude)

        if self._semiangle_cutoff == xp.inf:
            return xp.ones_like(alpha)

        if self._rolloff > 0.0:
            rolloff = self._rolloff / 1000.0  # * semiangle_cutoff
            array = 0.5 * (
                1 + xp.cos(np.pi * (alpha - semiangle_cutoff + rolloff) / rolloff)
            )
            array[alpha > semiangle_cutoff] = 0.0
            array = xp.where(
                alpha > semiangle_cutoff - rolloff,
                array,
                xp.ones_like(alpha, dtype=xp.float32),
            )
        else:
            array = xp.array(alpha < semiangle_cutoff).astype(xp.float32)
        return array

    def evaluate_temporal_envelope(
        self, alpha: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        xp = self._xp
        return xp.exp(
            -((0.5 * xp.pi / self._wavelength * self._focal_spread * alpha**2) ** 2)
        ).astype(xp.float32)

    def evaluate_gaussian_envelope(
        self, alpha: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        xp = self._xp
        return xp.exp(
            -0.5 * self._gaussian_spread**2 * alpha**2 / self._wavelength**2
        )

    def evaluate_spatial_envelope(
        self, alpha: Union[float, np.ndarray], phi: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        xp = self._xp
        p = self._parameters
        dchi_dk = (
            2
            * xp.pi
            / self._wavelength
            * (
                (p["C12"] * xp.cos(2.0 * (phi - p["phi12"])) + p["C10"]) * alpha
                + (
                    p["C23"] * xp.cos(3.0 * (phi - p["phi23"]))
                    + p["C21"] * xp.cos(1.0 * (phi - p["phi21"]))
                )
                * alpha**2
                + (
                    p["C34"] * xp.cos(4.0 * (phi - p["phi34"]))
                    + p["C32"] * xp.cos(2.0 * (phi - p["phi32"]))
                    + p["C30"]
                )
                * alpha**3
                + (
                    p["C45"] * xp.cos(5.0 * (phi - p["phi45"]))
                    + p["C43"] * xp.cos(3.0 * (phi - p["phi43"]))
                    + p["C41"] * xp.cos(1.0 * (phi - p["phi41"]))
                )
                * alpha**4
                + (
                    p["C56"] * xp.cos(6.0 * (phi - p["phi56"]))
                    + p["C54"] * xp.cos(4.0 * (phi - p["phi54"]))
                    + p["C52"] * xp.cos(2.0 * (phi - p["phi52"]))
                    + p["C50"]
                )
                * alpha**5
            )
        )

        dchi_dphi = (
            -2
            * xp.pi
            / self._wavelength
            * (
                1 / 2.0 * (2.0 * p["C12"] * xp.sin(2.0 * (phi - p["phi12"]))) * alpha
                + 1
                / 3.0
                * (
                    3.0 * p["C23"] * xp.sin(3.0 * (phi - p["phi23"]))
                    + 1.0 * p["C21"] * xp.sin(1.0 * (phi - p["phi21"]))
                )
                * alpha**2
                + 1
                / 4.0
                * (
                    4.0 * p["C34"] * xp.sin(4.0 * (phi - p["phi34"]))
                    + 2.0 * p["C32"] * xp.sin(2.0 * (phi - p["phi32"]))
                )
                * alpha**3
                + 1
                / 5.0
                * (
                    5.0 * p["C45"] * xp.sin(5.0 * (phi - p["phi45"]))
                    + 3.0 * p["C43"] * xp.sin(3.0 * (phi - p["phi43"]))
                    + 1.0 * p["C41"] * xp.sin(1.0 * (phi - p["phi41"]))
                )
                * alpha**4
                + 1
                / 6.0
                * (
                    6.0 * p["C56"] * xp.sin(6.0 * (phi - p["phi56"]))
                    + 4.0 * p["C54"] * xp.sin(4.0 * (phi - p["phi54"]))
                    + 2.0 * p["C52"] * xp.sin(2.0 * (phi - p["phi52"]))
                )
                * alpha**5
            )
        )

        return xp.exp(
            -xp.sign(self._angular_spread)
            * (self._angular_spread / 2 / 1000) ** 2
            * (dchi_dk**2 + dchi_dphi**2)
        )

    def evaluate_chi(
        self, alpha: Union[float, np.ndarray], phi: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        xp = self._xp
        p = self._parameters

        alpha2 = alpha**2
        alpha = xp.array(alpha)

        array = xp.zeros(alpha.shape, dtype=np.float32)
        if any([p[symbol] != 0.0 for symbol in ("C10", "C12", "phi12")]):
            array += (
                1 / 2 * alpha2 * (p["C10"] + p["C12"] * xp.cos(2 * (phi - p["phi12"])))
            )

        if any([p[symbol] != 0.0 for symbol in ("C21", "phi21", "C23", "phi23")]):
            array += (
                1
                / 3
                * alpha2
                * alpha
                * (
                    p["C21"] * xp.cos(phi - p["phi21"])
                    + p["C23"] * xp.cos(3 * (phi - p["phi23"]))
                )
            )

        if any(
            [p[symbol] != 0.0 for symbol in ("C30", "C32", "phi32", "C34", "phi34")]
        ):
            array += (
                1
                / 4
                * alpha2**2
                * (
                    p["C30"]
                    + p["C32"] * xp.cos(2 * (phi - p["phi32"]))
                    + p["C34"] * xp.cos(4 * (phi - p["phi34"]))
                )
            )

        if any(
            [
                p[symbol] != 0.0
                for symbol in ("C41", "phi41", "C43", "phi43", "C45", "phi41")
            ]
        ):
            array += (
                1
                / 5
                * alpha2**2
                * alpha
                * (
                    p["C41"] * xp.cos((phi - p["phi41"]))
                    + p["C43"] * xp.cos(3 * (phi - p["phi43"]))
                    + p["C45"] * xp.cos(5 * (phi - p["phi45"]))
                )
            )

        if any(
            [
                p[symbol] != 0.0
                for symbol in ("C50", "C52", "phi52", "C54", "phi54", "C56", "phi56")
            ]
        ):
            array += (
                1
                / 6
                * alpha2**3
                * (
                    p["C50"]
                    + p["C52"] * xp.cos(2 * (phi - p["phi52"]))
                    + p["C54"] * xp.cos(4 * (phi - p["phi54"]))
                    + p["C56"] * xp.cos(6 * (phi - p["phi56"]))
                )
            )

        array = 2 * xp.pi / self._wavelength * array + self._phase_shift
        return array

    def evaluate_aberrations(
        self, alpha: Union[float, np.ndarray], phi: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        xp = self._xp
        return xp.exp(-1.0j * self.evaluate_chi(alpha, phi))

    def evaluate(
        self, alpha: Union[float, np.ndarray], phi: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        array = self.evaluate_aberrations(alpha, phi)

        if self._semiangle_cutoff < np.inf or self._vacuum_probe_intensity is not None:
            array *= self.evaluate_aperture(alpha, phi)

        if self._focal_spread > 0.0:
            array *= self.evaluate_temporal_envelope(alpha)

        if self._angular_spread > 0.0:
            array *= self.evaluate_spatial_envelope(alpha, phi)

        if self._gaussian_spread > 0.0:
            array *= self.evaluate_gaussian_envelope(alpha)

        return array

    def _evaluate_ctf(self):
        alpha, phi = self.get_scattering_angles()

        array = self.evaluate(alpha, phi)
        return array

    def get_scattering_angles(self):
        kx, ky = self.get_spatial_frequencies()
        alpha, phi = self.polar_coordinates(
            kx * self._wavelength, ky * self._wavelength
        )
        return alpha, phi

    def get_spatial_frequencies(self):
        xp = self._xp
        kx, ky = spatial_frequencies(self._gpts, self._sampling)
        kx = xp.asarray(kx)
        ky = xp.asarray(ky)
        return kx, ky

    def polar_coordinates(self, x, y):
        """Calculate a polar grid for a given Cartesian grid."""
        xp = self._xp
        alpha = xp.sqrt(x.reshape((-1, 1)) ** 2 + y.reshape((1, -1)) ** 2)
        phi = xp.arctan2(x.reshape((-1, 1)), y.reshape((1, -1)))
        return alpha, phi

    def build(self):
        """Builds complex probe in the center of the region of interest."""
        xp = self._xp
        array = xp.fft.fftshift(xp.fft.ifft2(self._evaluate_ctf()))
        # if self._vacuum_probe_intensity is not None:
        array = array / xp.sqrt((xp.abs(array) ** 2).sum())
        self._array = array
        return self

    def visualize(self, **kwargs):
        """Plots the probe amplitude."""
        xp = self._xp
        asnumpy = self._asnumpy

        cmap = kwargs.get("cmap", "Greys_r")
        kwargs.pop("cmap", None)

        plt.imshow(
            asnumpy(xp.abs(self._array) ** 2),
            cmap=cmap,
            **kwargs,
        )
        return self


def spatial_frequencies(gpts: Tuple[int, int], sampling: Tuple[float, float]):
    """
    Calculate spatial frequencies of a grid.

    Parameters
    ----------
    gpts: tuple of int
        Number of grid points.
    sampling: tuple of float
        Sampling of the potential [1 / Å].

    Returns
    -------
    tuple of arrays
    """

    return tuple(
        np.fft.fftfreq(n, d).astype(np.float32) for n, d in zip(gpts, sampling)
    )


def projection(u: np.ndarray, v: np.ndarray, xp):
    """Projection of vector u onto vector v."""
    return u * xp.vdot(u, v) / xp.vdot(u, u)


def orthogonalize(V: np.ndarray, xp):
    """Non-normalized QR decomposition using repeated projections."""
    U = V.copy()
    for i in range(1, V.shape[0]):
        for j in range(i):
            U[i, :] -= projection(U[j, :], V[i, :], xp)
    return U


### FFT-shift functions


def fourier_translation_operator(
    positions: np.ndarray, shape: tuple, xp=np
) -> np.ndarray:
    """
    Create an array representing one or more phase ramp(s) for shifting another array.

    Parameters
    ----------
    positions : array of xy-positions
        Positions to calculate fourier translation operators for
    shape : two int
        Array dimensions to be fourier-shifted
    xp: Callable
        Array computing module

    Returns
    -------
    Fourier translation operators
    """

    positions_shape = positions.shape

    if len(positions_shape) == 1:
        positions = positions[None]

    kx, ky = spatial_frequencies(shape, (1.0, 1.0))
    kx = kx.reshape((1, -1, 1))
    ky = ky.reshape((1, 1, -1))
    kx = xp.asarray(kx)
    ky = xp.asarray(ky)
    positions = xp.asarray(positions)
    x = positions[:, 0].reshape((-1,) + (1, 1))
    y = positions[:, 1].reshape((-1,) + (1, 1))

    result = xp.exp(-2.0j * np.pi * kx * x) * xp.exp(-2.0j * np.pi * ky * y)

    if len(positions_shape) == 1:
        return result[0]
    else:
        return result


def fft_shift(array, positions, xp=np):
    """
    Fourier-shift array using positions.

    Parameters
    ----------
    array: np.ndarray
        Array to be shifted
    positions: array of xy-positions
        Positions to fourier-shift array with
    xp: Callable
        Array computing module

    Returns
    -------
        Fourier-shifted array
    """
    translation_operator = fourier_translation_operator(positions, array.shape[-2:], xp)
    fourier_array = xp.fft.fft2(array)

    if len(translation_operator.shape) == 3 and len(fourier_array.shape) == 3:
        shifted_fourier_array = fourier_array[None] * translation_operator[:, None]
    else:
        shifted_fourier_array = fourier_array * translation_operator

    return xp.fft.ifft2(shifted_fourier_array)


### Standalone DPC functions


def calculate_center_of_mass(
    intensities: np.ndarray,
    fit_function: str = "plane",
    plot_center_of_mass: bool = True,
    scan_sampling: Tuple[float, float] = (1.0, 1.0),
    reciprocal_sampling: Tuple[float, float] = (1.0, 1.0),
    scan_units: Tuple[str, str] = ("pixels", "pixels"),
    device: str = "cpu",
    **kwargs,
):
    """
    Common preprocessing function to compute and fit diffraction intensities CoM

    Parameters
    ----------
    intensities: (Rx,Ry,Qx,Qy) xp.ndarray
        Raw intensities array stored on device, with dtype xp.float32
    fit_function: str, optional
        2D fitting function for CoM fitting. One of 'plane','parabola','bezier_two'
    plot_center_of_mass: bool, optional
        If True, the computed and normalized CoM arrays will be displayed
    scan_sampling: Tuple[float,float], optional
        Real-space scan sampling in `scan_units`
    reciprocal_sampling: Tuple[float,float], optional
        Reciprocal-space sampling in `A^-1`
    scan_units: Tuple[str,str], optional
        Real-space scan sampling units
    device: str, optional
        Device to perform calculations on. Must be either 'cpu' or 'gpu'

    Returns
    --------
    com_normalized_x: (Rx,Ry) xp.ndarray
        Normalized horizontal center of mass gradient
    com_normalized_y: (Rx,Ry) xp.ndarray
        Normalized vertical center of mass gradient

    Displays
    --------
    com_measured_x/y and com_normalized_x/y, optional
        Measured and normalized CoM gradients
    """

    if device == "cpu":
        xp = np
        asnumpy = np.asarray
        if isinstance(intensities, np.ndarray):
            intensities = asnumpy(intensities)
        else:
            intensities = cp.asnumpy(intensities)
    elif device == "gpu":
        xp = cp
        asnumpy = cp.asnumpy
        intensities = xp.asarray(intensities)
    else:
        raise ValueError(f"device must be either 'cpu' or 'gpu', not {device}")

    intensities_shape = np.array(intensities.shape)
    intensities_sum = xp.sum(intensities, axis=(-2, -1))

    # Coordinates
    kx = xp.arange(intensities_shape[-2], dtype=xp.float32)
    ky = xp.arange(intensities_shape[-1], dtype=xp.float32)
    kya, kxa = xp.meshgrid(ky, kx)

    # calculate CoM
    com_measured_x = (
        xp.sum(intensities * kxa[None, None], axis=(-2, -1)) / intensities_sum
    )
    com_measured_y = (
        xp.sum(intensities * kya[None, None], axis=(-2, -1)) / intensities_sum
    )

    # Fit function to center of mass
    # TO-DO: allow py4DSTEM.process.calibration.fit_origin to accept xp.ndarrays
    or_fits = fit_origin(
        (asnumpy(com_measured_x), asnumpy(com_measured_y)),
        fitfunction=fit_function,
    )
    com_fitted_x = xp.asarray(or_fits[0])
    com_fitted_y = xp.asarray(or_fits[1])

    # fix CoM units
    com_normalized_x = (com_measured_x - com_fitted_x) * reciprocal_sampling[0]
    com_normalized_y = (com_measured_y - com_fitted_y) * reciprocal_sampling[1]

    # Optionally, plot
    if plot_center_of_mass:

        figsize = kwargs.get("figsize", (8, 8))
        cmap = kwargs.get("cmap", "RdBu_r")
        kwargs.pop("cmap", None)
        kwargs.pop("figsize", None)

        extent = [
            0,
            scan_sampling[1] * intensities_shape[1],
            scan_sampling[0] * intensities_shape[0],
            0,
        ]

        fig = plt.figure(figsize=figsize)
        grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=(0.25, 0.5))

        for ax, arr, title in zip(
            grid,
            [
                com_measured_x,
                com_measured_y,
                com_normalized_x,
                com_normalized_y,
            ],
            ["CoM_x", "CoM_y", "Normalized CoM_x", "Normalized CoM_y"],
        ):
            ax.imshow(asnumpy(arr), extent=extent, cmap=cmap, **kwargs)
            ax.set_xlabel(f"x [{scan_units[0]}]")
            ax.set_ylabel(f"y [{scan_units[1]}]")
            ax.set_title(title)

    return asnumpy(com_normalized_x), asnumpy(com_normalized_y)


def center_of_mass_relative_rotation(
    com_normalized_x: np.ndarray,
    com_normalized_y: np.ndarray,
    rotation_angles_deg: np.ndarray = np.arange(-89.0, 90.0, 1.0),
    plot_rotation: bool = True,
    maximize_divergence: bool = False,
    device: str = "cpu",
    **kwargs,
):
    """
    Solves for the relative rotation between scan directions
    and the reciprocal coordinate system. We do this by minimizing the curl of the
    CoM gradient vector field or, alternatively, maximizing the divergence.

    Parameters
    ----------
    com_normalized_x: (Rx,Ry) xp.ndarray
        Normalized horizontal center of mass gradient
    com_normalized_y: (Rx,Ry) xp.ndarray
        Normalized vertical center of mass gradient
    rotation_angles_deg: ndarray, optional
        Array of angles in degrees to perform curl minimization over
    plot_rotation: bool, optional
        If True, the CoM curl minimization search result will be displayed
    maximize_divergence: bool, optional
        If True, the divergence of the CoM gradient vector field is maximized
    device: str, optional
        Device to perform calculations on. Must be either 'cpu' or 'gpu'

    Returns
    --------
    self.com_x: np.ndarray
        Corrected horizontal center of mass gradient, as a numpy array
    self.com_y: np.ndarray
        Corrected vertical center of mass gradient, as a numpy array
    rotation_best_deg: float
        Rotation angle which minimizes CoM curl, in degrees
    rotation_best_transpose: bool
        Whether diffraction intensities need to be transposed to minimize CoM curl

    Displays
    --------
    rotation_curl/div vs rotation_angles_deg, optional
        Vector calculus quantity being minimized/maximized
    rotation_best_deg
        Summary statistics
    """

    if device == "cpu":
        xp = np
        asnumpy = np.asarray
        if isinstance(com_normalized_x, np.ndarray):
            com_normalized_x = asnumpy(com_normalized_x)
        else:
            com_normalized_x = cp.asnumpy(com_normalized_x)
        if isinstance(com_normalized_y, np.ndarray):
            com_normalized_y = asnumpy(com_normalized_y)
        else:
            com_normalized_y = cp.asnumpy(com_normalized_y)
    elif device == "gpu":
        xp = cp
        asnumpy = cp.asnumpy
        com_normalized_x = xp.asarray(com_normalized_x)
        com_normalized_y = xp.asarray(com_normalized_y)
    else:
        raise ValueError(f"device must be either 'cpu' or 'gpu', not {device}")

    rotation_angles_deg = xp.asarray(rotation_angles_deg)
    rotation_angles_rad = xp.deg2rad(rotation_angles_deg)[:, None, None]

    # Untransposed
    com_measured_x = (
        xp.cos(rotation_angles_rad) * com_normalized_x[None]
        - xp.sin(rotation_angles_rad) * com_normalized_y[None]
    )
    com_measured_y = (
        xp.sin(rotation_angles_rad) * com_normalized_x[None]
        + xp.cos(rotation_angles_rad) * com_normalized_y[None]
    )

    if maximize_divergence:
        com_grad_x_x = com_measured_x[:, 2:, 1:-1] - com_measured_x[:, :-2, 1:-1]
        com_grad_y_y = com_measured_y[:, 1:-1, 2:] - com_measured_y[:, 1:-1, :-2]
        rotation_div = xp.mean(xp.abs(com_grad_x_x + com_grad_y_y), axis=(-2, -1))
    else:
        com_grad_x_y = com_measured_x[:, 1:-1, 2:] - com_measured_x[:, 1:-1, :-2]
        com_grad_y_x = com_measured_y[:, 2:, 1:-1] - com_measured_y[:, :-2, 1:-1]
        rotation_curl = xp.mean(xp.abs(com_grad_y_x - com_grad_x_y), axis=(-2, -1))

    # Transposed
    com_measured_x = (
        xp.cos(rotation_angles_rad) * com_normalized_y[None]
        - xp.sin(rotation_angles_rad) * com_normalized_x[None]
    )
    com_measured_y = (
        xp.sin(rotation_angles_rad) * com_normalized_y[None]
        + xp.cos(rotation_angles_rad) * com_normalized_x[None]
    )

    if maximize_divergence:
        com_grad_x_x = com_measured_x[:, 2:, 1:-1] - com_measured_x[:, :-2, 1:-1]
        com_grad_y_y = com_measured_y[:, 1:-1, 2:] - com_measured_y[:, 1:-1, :-2]
        rotation_div_transpose = xp.mean(
            xp.abs(com_grad_x_x + com_grad_y_y), axis=(-2, -1)
        )
    else:
        com_grad_x_y = com_measured_x[:, 1:-1, 2:] - com_measured_x[:, 1:-1, :-2]
        com_grad_y_x = com_measured_y[:, 2:, 1:-1] - com_measured_y[:, :-2, 1:-1]
        rotation_curl_transpose = xp.mean(
            xp.abs(com_grad_y_x - com_grad_x_y), axis=(-2, -1)
        )

    rotation_angles_rad = asnumpy(xp.squeeze(rotation_angles_rad))
    rotation_angles_deg = asnumpy(rotation_angles_deg)

    # Find lowest curl/ maximum div value
    if maximize_divergence:
        # Maximize Divergence
        ind_max = xp.argmax(rotation_div).item()
        ind_trans_max = xp.argmax(rotation_div_transpose).item()

        if rotation_div[ind_max] >= rotation_div_transpose[ind_trans_max]:
            rotation_best_deg = rotation_angles_deg[ind_max]
            rotation_best_rad = rotation_angles_rad[ind_max]
            rotation_best_transpose = False
        else:
            rotation_best_deg = rotation_angles_deg[ind_trans_max]
            rotation_best_rad = rotation_angles_rad[ind_trans_max]
            rotation_best_transpose = True
    else:
        # Minimize Curl
        ind_min = xp.argmin(rotation_curl).item()
        ind_trans_min = xp.argmin(rotation_curl_transpose).item()

        if rotation_curl[ind_min] <= rotation_curl_transpose[ind_trans_min]:
            rotation_best_deg = rotation_angles_deg[ind_min]
            rotation_best_rad = rotation_angles_rad[ind_min]
            rotation_best_transpose = False
        else:
            rotation_best_deg = rotation_angles_deg[ind_trans_min]
            rotation_best_rad = rotation_angles_rad[ind_trans_min]
            rotation_best_transpose = True

    # Print summary
    print(("Best fit rotation = " f"{str(np.round(rotation_best_deg))} degrees."))
    if rotation_best_transpose:
        print("Diffraction intensities should be transposed.")
    else:
        print("No need to transpose diffraction intensities.")

    # Plot Curl/Div rotation
    if plot_rotation:

        figsize = kwargs.get("figsize", (8, 2))
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            rotation_angles_deg,
            asnumpy(rotation_div) if maximize_divergence else asnumpy(rotation_curl),
            label="CoM",
        )
        ax.plot(
            rotation_angles_deg,
            asnumpy(rotation_div_transpose)
            if maximize_divergence
            else asnumpy(rotation_curl_transpose),
            label="CoM after transpose",
        )
        y_r = ax.get_ylim()
        ax.plot(
            np.ones(2) * rotation_best_deg,
            y_r,
            color=(0, 0, 0, 1),
        )

        ax.legend(loc="best")
        ax.set_xlabel("Rotation [degrees]")
        if maximize_divergence:
            ax.set_ylabel("Mean Absolute Divergence")
            ax.set_aspect(
                np.ptp(rotation_angles_deg)
                / np.maximum(
                    np.ptp(rotation_div),
                    np.ptp(rotation_div_transpose),
                )
                / 4
            )
        else:
            ax.set_ylabel("Mean Absolute Curl")
            ax.set_aspect(
                np.ptp(rotation_angles_deg)
                / np.maximum(
                    np.ptp(rotation_curl),
                    np.ptp(rotation_curl_transpose),
                )
                / 4
            )
        fig.tight_layout()

    # Calculate corrected CoM
    if rotation_best_transpose:
        com_x = (
            xp.cos(rotation_best_rad) * com_normalized_y
            - xp.sin(rotation_best_rad) * com_normalized_x
        )
        com_y = (
            xp.sin(rotation_best_rad) * com_normalized_y
            + xp.cos(rotation_best_rad) * com_normalized_x
        )
    else:
        com_x = (
            xp.cos(rotation_best_rad) * com_normalized_x
            - xp.sin(rotation_best_rad) * com_normalized_y
        )
        com_y = (
            xp.sin(rotation_best_rad) * com_normalized_x
            + xp.cos(rotation_best_rad) * com_normalized_y
        )

    com_x = asnumpy(com_x)
    com_y = asnumpy(com_y)

    return com_x, com_y, rotation_best_deg, rotation_best_transpose


### Batching functions


def subdivide_into_batches(
    num_items: int, num_batches: int = None, max_batch: int = None
):
    """
    Split an n integer into m (almost) equal integers, such that the sum of smaller integers equals n.

    Parameters
    ----------
    n: int
        The integer to split.
    m: int
        The number integers n will be split into.

    Returns
    -------
    list of int
    """
    if (num_batches is not None) & (max_batch is not None):
        raise RuntimeError()

    if num_batches is None:
        if max_batch is not None:
            num_batches = (num_items + (-num_items % max_batch)) // max_batch
        else:
            raise RuntimeError()

    if num_items < num_batches:
        raise RuntimeError("num_batches may not be larger than num_items")

    elif num_items % num_batches == 0:
        return [num_items // num_batches] * num_batches
    else:
        v = []
        zp = num_batches - (num_items % num_batches)
        pp = num_items // num_batches
        for i in range(num_batches):
            if i >= zp:
                v = [pp + 1] + v
            else:
                v = [pp] + v
        return v


def generate_batches(
    num_items: int, num_batches: int = None, max_batch: int = None, start=0
):
    for batch in subdivide_into_batches(num_items, num_batches, max_batch):
        end = start + batch
        yield start, end

        start = end


#### Affine transformation functions
# Adapted from https://github.com/AdvancedPhotonSource/tike/blob/f9004a32fda5e49fa63b987e9ffe3c8447d59950/src/tike/ptycho/position.py


class AffineTransform:
    """
    Affine Transform Class.

    Simplified version of AffineTransform from tike:
    https://github.com/AdvancedPhotonSource/tike/blob/f9004a32fda5e49fa63b987e9ffe3c8447d59950/src/tike/ptycho/position.py

    AffineTransform() -> Identity

    Parameters
    ----------
    scale0: float
        x-scaling
    scale1: float
        y-scaling
    shear1: float
        \gamma shear
    angle: float
        \theta rotation angle
    t0: float
        x-translation
    t1: float
        y-translation
    """

    def __init__(
        self,
        scale0: float = 1.0,
        scale1: float = 1.0,
        shear1: float = 0.0,
        angle: float = 0.0,
        t0: float = 0.0,
        t1: float = 0.0,
    ):

        self.scale0 = scale0
        self.scale1 = scale1
        self.shear1 = shear1
        self.angle = angle
        self.t0 = t0
        self.t1 = t1

    @classmethod
    def fromarray(self, T: np.ndarray):
        """Return an Affine Transfrom from a 2x2 matrix.
        Use decomposition method from Graphics Gems 2 Section 7.1
        """
        R = T[:2, :2].copy()
        scale0 = np.linalg.norm(R[0])
        if scale0 <= 0:
            return AffineTransform()
        R[0] /= scale0
        shear1 = R[0] @ R[1]
        R[1] -= shear1 * R[0]
        scale1 = np.linalg.norm(R[1])
        if scale1 <= 0:
            return AffineTransform()
        R[1] /= scale1
        shear1 /= scale1
        angle = np.arccos(R[0, 0])

        if T.shape[0] > 2:
            t0, t1 = T[2]
        else:
            t0 = t1 = 0.0

        return AffineTransform(
            scale0=float(scale0),
            scale1=float(scale1),
            shear1=float(shear1),
            angle=float(angle),
            t0=t0,
            t1=t1,
        )

    def asarray(self):
        """Return an 2x2 matrix of scale, shear, rotation.
        This matrix is scale @ shear @ rotate from left to right.
        """
        cosx = np.cos(self.angle)
        sinx = np.sin(self.angle)
        return (
            np.array(
                [
                    [self.scale0, 0.0],
                    [0.0, self.scale1],
                ],
                dtype="float32",
            )
            @ np.array(
                [
                    [1.0, 0.0],
                    [self.shear1, 1.0],
                ],
                dtype="float32",
            )
            @ np.array(
                [
                    [+cosx, -sinx],
                    [+sinx, +cosx],
                ],
                dtype="float32",
            )
        )

    def asarray3(self):
        """Return an 3x2 matrix of scale, shear, rotation, translation.
        This matrix is scale @ shear @ rotate from left to right. Expects a
        homogenous (z) coordinate of 1.
        """
        T = np.empty((3, 2), dtype="float32")
        T[2] = (self.t0, self.t1)
        T[:2, :2] = self.asarray()
        return T

    def astuple(self):
        """Return the constructor parameters in a tuple."""
        return (
            self.scale0,
            self.scale1,
            self.shear1,
            self.angle,
            self.t0,
            self.t1,
        )

    def __call__(self, x: np.ndarray, origin=(0, 0), xp=np):
        origin = xp.asarray(origin)
        tf_matrix = self.asarray()
        tf_matrix = xp.asarray(tf_matrix)
        tf_translation = xp.array((self.t0, self.t1)) + origin
        return ((x - origin) @ tf_matrix) + tf_translation

    def __str__(self):
        return (
            "AffineTransform( \n"
            f"  scale0 = {self.scale0:.4f}, scale1 = {self.scale1:.4f}, \n"
            f"  shear1 = {self.shear1:.4f}, angle = {self.angle:.4f}, \n"
            f"  t0 = {self.t0:.4f}, t1 = {self.t1:.4f}, \n"
            ")"
        )


def estimate_global_transformation(
    positions0: np.ndarray,
    positions1: np.ndarray,
    origin: Tuple[int, int] = (0, 0),
    translation_allowed: bool = True,
    xp=np,
):
    """Use least squares to estimate the global affine transformation."""

    origin = xp.asarray(origin)

    try:
        if translation_allowed:
            a = xp.pad(positions0 - origin, ((0, 0), (0, 1)), constant_values=1)
        else:
            a = positions0 - origin

        b = positions1 - origin
        aT = a.conj().swapaxes(-1, -2)
        x = xp.linalg.inv(aT @ a) @ aT @ b

        tf = AffineTransform.fromarray(x)

    except xp.linalg.LinAlgError:
        tf = AffineTransform()

    error = xp.linalg.norm(tf(positions0, origin=origin, xp=xp) - positions1)

    return tf, error


def estimate_global_transformation_ransac(
    positions0: np.ndarray,
    positions1: np.ndarray,
    origin: Tuple[int, int] = (0, 0),
    translation_allowed: bool = True,
    min_sample: int = 64,
    max_error: float = 16,
    min_consensus: float = 0.75,
    max_iter: int = 20,
    xp=np,
):
    """Use RANSAC to estimate the global affine transformation."""
    best_fitness = np.inf  # small fitness is good
    transform = AffineTransform()

    # Choose a subset
    for subset in np.random.choice(
        a=len(positions0),
        size=(max_iter, min_sample),
        replace=True,
    ):
        # Fit to subset
        subset = np.unique(subset)
        candidate_model, _ = estimate_global_transformation(
            positions0=positions0[subset],
            positions1=positions1[subset],
            origin=origin,
            translation_allowed=translation_allowed,
            xp=xp,
        )

        # Determine inliars and outliars
        position_error = xp.linalg.norm(
            candidate_model(positions0, origin=origin, xp=xp) - positions1,
            axis=-1,
        )
        inliars = position_error <= max_error

        # Check if consensus reached
        if xp.sum(inliars) / len(inliars) >= min_consensus:
            # Refit with consensus inliars
            candidate_model, fitness = estimate_global_transformation(
                positions0=positions0[inliars],
                positions1=positions1[inliars],
                origin=origin,
                translation_allowed=translation_allowed,
                xp=xp,
            )
            if fitness < best_fitness:
                best_fitness = fitness
                transform = candidate_model

    return transform, best_fitness


def fourier_ring_correlation(
    image_1,
    image_2,
    pixel_size=None,
    bin_size=None,
    sigma=None,
    align_images=False,
    upsample_factor=8,
    device="cpu",
    plot_frc=True,
    frc_color="red",
    half_bit_color="blue",
):
    """
    Computes fourier ring correlation (FRC) of 2 arrays.
    Arrays must bet the same size.

    Parameters
     ----------
    image1: ndarray
        first image for FRC
    image2: ndarray
        second image for FRC
    pixel_size: tuple
        size of pixels in A (x,y)
    bin_size: float, optional
        size of bins for ring profile
    sigma: float, optional
        standard deviation for Gaussian kernel
    align_images: bool
        if True, aligns images using DFT upsampling of cross correlation.
    upsample factor: int
        if align_images, upsampling for correlation. Must be greater than 2.
    device: str, optional
        calculation device will be perfomed on. Must be 'cpu' or 'gpu'
    plot_frc: bool, optional
        if True, plots frc
    frc_color: str, optional
        color of FRC line in plot
    half_bit_color: str, optional
        color of half-bit line

    Returns
    --------
    q_frc: ndarray
        spatial frequencies of FRC
    frc: ndarray
        fourier ring correlation
    half_bit: ndarray
        half-bit criteria
    """

    if align_images:
        from py4DSTEM.process.utils.cross_correlate import align_and_shift_images

        image_2 = align_and_shift_images(
            image_1,
            image_2,
            upsample_factor=upsample_factor,
            device=device,
        )

    if device == "cpu":
        xp = np

    elif device == "gpu":
        xp = cp

    fft_image_1 = xp.fft.fft2(image_1)
    fft_image_2 = xp.fft.fft2(image_2)

    cc_mixed = xp.real(fft_image_1 * xp.conj(fft_image_2))
    cc_image_1 = xp.abs(fft_image_1) ** 2
    cc_image_2 = xp.abs(fft_image_2) ** 2

    # take 1D profile
    q_frc, cc_mixed_1D, n = return_1D_profile(
        cc_mixed,
        pixel_size=pixel_size,
        sigma=sigma,
        bin_size=bin_size,
        device=device,
    )
    _, cc_image_1_1D, _ = return_1D_profile(
        cc_image_1, pixel_size=pixel_size, sigma=sigma, bin_size=bin_size, device=device
    )
    _, cc_image_2_1D, _ = return_1D_profile(
        cc_image_2,
        pixel_size=pixel_size,
        sigma=sigma,
        bin_size=bin_size,
        device=device,
    )

    frc = cc_mixed_1D / ((cc_image_1_1D * cc_image_2_1D) ** 0.5)
    half_bit = 2 / xp.sqrt(n / 2)

    ind_max = xp.argmax(n)
    q_frc = q_frc[1:ind_max]
    frc = frc[1:ind_max]
    half_bit = half_bit[1:ind_max]

    if plot_frc:
        fig, ax = plt.subplots()
        if device == "gpu":
            ax.plot(q_frc.get(), frc.get(), label="FRC", color=frc_color)
            ax.plot(q_frc.get(), half_bit.get(), label="half bit", color=half_bit_color)
            ax.set_xlim([0, q_frc.get().max()])
        else:
            ax.plot(q_frc, frc, label="FRC", color=frc_color)
            ax.plot(q_frc, half_bit, label="half bit", color=half_bit_color)
            ax.set_xlim([0, q_frc.max()])
        ax.legend()
        ax.set_ylim([0, 1])

        if pixel_size is None:
            ax.set_xlabel(r"Spatial frequency (pixels)")
        else:
            ax.set_xlabel(r"Spatial frequency ($\AA$)")
        ax.set_ylabel("FRC")

    return q_frc, frc, half_bit


def return_1D_profile(
    intensity, pixel_size=None, bin_size=None, sigma=None, device="cpu"
):
    """
    Return 1D radial profile from corner centered array

    Parameters
     ----------
    intensity: ndarray
        Array for computing 1D profile
    pixel_size: tuple
        Size of pixels in A (x,y)
    bin_size: float, optional
        Size of bins for ring profile
    sigma: float, optional
        standard deviation for Gaussian kernel
    device: str, optional
        calculation device will be perfomed on. Must be 'cpu' or 'gpu'

    Returns
    --------
    q_bins: ndarray
        spatial frequencies of bins
    I_bins: ndarray
        Intensity of bins
    n: ndarray
        Number of pixels in each bin
    """
    if pixel_size is None:
        pixel_size = (1, 1)

    if device == "cpu":
        xp = np

    elif device == "gpu":
        xp = cp

    x = xp.fft.fftfreq(intensity.shape[0], pixel_size[0])
    y = xp.fft.fftfreq(intensity.shape[1], pixel_size[1])
    q = xp.sqrt(x[:, None] ** 2 + y[None, :] ** 2)
    q = q.ravel()

    intensity = intensity.ravel()

    if bin_size is None:
        bin_size = q[1] - q[0]

    q_bins = xp.arange(0, q.max() + bin_size, bin_size)

    inds = q / bin_size
    inds_f = xp.floor(inds).astype("int")
    d_ind = inds - inds_f

    nf = xp.bincount(inds_f, weights=(1 - d_ind), minlength=q_bins.shape[0])
    nc = xp.bincount(inds_f + 1, weights=(d_ind), minlength=q_bins.shape[0])
    n = nf + nc

    I_bins0 = xp.bincount(
        inds_f, weights=intensity * (1 - d_ind), minlength=q_bins.shape[0]
    )
    I_bins1 = xp.bincount(
        inds_f + 1, weights=intensity * (d_ind), minlength=q_bins.shape[0]
    )

    I_bins = (I_bins0 + I_bins1) / n
    if sigma is not None:
        I_bins = gaussian_filter(I_bins, sigma)

    return q_bins, I_bins, n
