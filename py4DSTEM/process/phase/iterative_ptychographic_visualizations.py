import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from py4DSTEM.process.phase.utils import AffineTransform
from py4DSTEM.visualize import return_scaled_histogram_ordering, show, show_complex
from scipy.ndimage import gaussian_filter, rotate

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = np

warnings.simplefilter(action="always", category=UserWarning)


class VisualizationsMixin:
    """
    Mixin class for various visualization methods.
    """

    def show_updated_positions(
        self,
        scale_arrows=1,
        plot_arrow_freq=None,
        plot_cropped_rotated_fov=True,
        cbar=True,
        verbose=True,
        **kwargs,
    ):
        """
        Function to plot changes to probe positions during ptychography reconstruciton

        Parameters
        ----------
        scale_arrows: float, optional
            scaling factor to be applied on vectors prior to plt.quiver call
        plot_arrow_freq: int, optional
            thinning parameter to only plot a subset of probe positions
            assumes grid position
        verbose: bool, optional
            if True, prints AffineTransformation if positions have been updated
        """

        if verbose:
            if hasattr(self, "_tf"):
                print(self._tf)

        asnumpy = self._asnumpy

        initial_pos = asnumpy(self._positions_initial)
        pos = self.positions

        if plot_cropped_rotated_fov:
            angle = (
                self._rotation_best_rad
                if self._rotation_best_transpose
                else -self._rotation_best_rad
            )

            tf = AffineTransform(angle=angle)
            initial_pos = tf(initial_pos, origin=np.mean(pos, axis=0))
            pos = tf(pos, origin=np.mean(pos, axis=0))

            obj_shape = self.object_cropped.shape[-2:]
            initial_pos_com = np.mean(initial_pos, axis=0)
            center_shift = initial_pos_com - (
                np.array(obj_shape) / 2 * np.array(self.sampling)
            )
            initial_pos -= center_shift
            pos -= center_shift

        else:
            obj_shape = self._object_shape

        if plot_arrow_freq is not None:
            rshape = self._datacube.Rshape + (2,)
            freq = plot_arrow_freq

            initial_pos = initial_pos.reshape(rshape)[::freq, ::freq].reshape(-1, 2)
            pos = pos.reshape(rshape)[::freq, ::freq].reshape(-1, 2)

        deltas = pos - initial_pos
        norms = np.linalg.norm(deltas, axis=1)

        extent = [
            0,
            self.sampling[1] * obj_shape[1],
            self.sampling[0] * obj_shape[0],
            0,
        ]

        figsize = kwargs.pop("figsize", (6, 6))
        cmap = kwargs.pop("cmap", "Reds")

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.quiver(
            initial_pos[:, 1],
            initial_pos[:, 0],
            deltas[:, 1] * scale_arrows,
            deltas[:, 0] * scale_arrows,
            norms,
            scale_units="xy",
            scale=1,
            cmap=cmap,
            **kwargs,
        )

        if cbar:
            divider = make_axes_locatable(ax)
            ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
            fig.add_axes(ax_cb)
            cb = fig.colorbar(im, cax=ax_cb)
            cb.set_label("Δ [A]", rotation=0, ha="left", va="bottom")
            cb.ax.yaxis.set_label_coords(0.5, 1.01)

        ax.set_ylabel("x [A]")
        ax.set_xlabel("y [A]")
        ax.set_xlim((extent[0], extent[1]))
        ax.set_ylim((extent[2], extent[3]))
        ax.set_aspect("equal")
        ax.set_title("Updated probe positions")

    def show_uncertainty_visualization(
        self,
        errors=None,
        max_batch_size=None,
        projected_cropped_potential=None,
        kde_sigma=None,
        plot_histogram=True,
        plot_contours=False,
        **kwargs,
    ):
        """Plot uncertainty visualization using self-consistency errors"""

        xp = self._xp
        asnumpy = self._asnumpy
        gaussian_filter = self._gaussian_filter

        if errors is None:
            errors = self._return_self_consistency_errors(max_batch_size=max_batch_size)
        errors_xp = xp.asarray(errors)

        if projected_cropped_potential is None:
            projected_cropped_potential = self._return_projected_cropped_potential()

        if kde_sigma is None:
            kde_sigma = 0.5 * self._scan_sampling[0] / self.sampling[0]

        ## Kernel Density Estimation

        # rotated basis
        angle = (
            self._rotation_best_rad
            if self._rotation_best_transpose
            else -self._rotation_best_rad
        )

        tf = AffineTransform(angle=angle)
        rotated_points = tf(self._positions_px, origin=self._positions_px_com, xp=xp)

        padding = xp.min(rotated_points, axis=0).astype("int")

        # bilinear sampling
        pixel_output = np.array(projected_cropped_potential.shape) + asnumpy(
            2 * padding
        )
        pixel_size = pixel_output.prod()

        xa = rotated_points[:, 0]
        ya = rotated_points[:, 1]

        # bilinear sampling
        xF = xp.floor(xa).astype("int")
        yF = xp.floor(ya).astype("int")
        dx = xa - xF
        dy = ya - yF

        # resampling
        all_inds = [
            [xF, yF],
            [xF + 1, yF],
            [xF, yF + 1],
            [xF + 1, yF + 1],
        ]

        all_weights = [
            (1 - dx) * (1 - dy),
            (dx) * (1 - dy),
            (1 - dx) * (dy),
            (dx) * (dy),
        ]

        pix_count = xp.zeros(pixel_size, dtype=xp.float32)
        pix_output = xp.zeros(pixel_size, dtype=xp.float32)

        for inds, weights in zip(all_inds, all_weights):
            inds_1D = xp.ravel_multi_index(
                inds,
                pixel_output,
                mode=["wrap", "wrap"],
            )

            pix_count += xp.bincount(
                inds_1D,
                weights=weights,
                minlength=pixel_size,
            )
            pix_output += xp.bincount(
                inds_1D,
                weights=weights * errors_xp,
                minlength=pixel_size,
            )

        # reshape 1D arrays to 2D
        pix_count = xp.reshape(
            pix_count,
            pixel_output,
        )
        pix_output = xp.reshape(
            pix_output,
            pixel_output,
        )

        # kernel density estimate
        pix_count = gaussian_filter(pix_count, kde_sigma)
        pix_output = gaussian_filter(pix_output, kde_sigma)
        sub = pix_count > 1e-3
        pix_output[sub] /= pix_count[sub]
        pix_output[np.logical_not(sub)] = 1
        pix_output = pix_output[padding[0] : -padding[0], padding[1] : -padding[1]]
        pix_output, _, _ = return_scaled_histogram_ordering(
            pix_output.get(), normalize=True
        )

        ## Visualization
        if plot_histogram:
            spec = GridSpec(
                ncols=1,
                nrows=2,
                height_ratios=[1, 4],
                hspace=0.15,
            )
            auto_figsize = (4, 5.25)
        else:
            spec = GridSpec(
                ncols=1,
                nrows=1,
            )
            auto_figsize = (4, 4)

        figsize = kwargs.pop("figsize", auto_figsize)

        fig = plt.figure(figsize=figsize)

        if plot_histogram:
            ax_hist = fig.add_subplot(spec[0])

            counts, bins = np.histogram(errors, bins=50)
            ax_hist.hist(bins[:-1], bins, weights=counts, color="#5ac8c8", alpha=0.5)
            ax_hist.set_ylabel("Counts")
            ax_hist.set_xlabel("Normalized Squared Error")

        ax = fig.add_subplot(spec[-1])

        cmap = kwargs.pop("cmap", "magma")
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)

        projected_cropped_potential, vmin, vmax = return_scaled_histogram_ordering(
            projected_cropped_potential,
            vmin=vmin,
            vmax=vmax,
        )

        extent = [
            0,
            self.sampling[1] * projected_cropped_potential.shape[1],
            self.sampling[0] * projected_cropped_potential.shape[0],
            0,
        ]

        ax.imshow(
            projected_cropped_potential,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            alpha=1 - pix_output,
            cmap=cmap,
            **kwargs,
        )

        if plot_contours:
            aligned_points = asnumpy(rotated_points - padding)
            aligned_points[:, 0] *= self.sampling[0]
            aligned_points[:, 1] *= self.sampling[1]

            ax.tricontour(
                aligned_points[:, 1],
                aligned_points[:, 0],
                errors,
                colors="grey",
                levels=5,
                # linestyles='dashed',
                linewidths=0.5,
            )

        ax.set_ylabel("x [A]")
        ax.set_xlabel("y [A]")
        ax.set_xlim((extent[0], extent[1]))
        ax.set_ylim((extent[2], extent[3]))
        ax.xaxis.set_ticks_position("bottom")

        spec.tight_layout(fig)
