from ...file.datastructure import DataCube
from . import WPFModelPrototype

from typing import Optional
import numpy as np

from scipy.optimize import least_squares


class WholePatternFit:
    def __init__(
        self,
        datacube: DataCube,
        x0: Optional[float] = None,
        y0: Optional[float] = None,
        mask: Optional[np.ndarray] = None,
        use_jacobian: bool = True,
        meanCBED: Optional[np.ndarray] = None,
    ):
        self.datacube = datacube
        self.meanCBED = (
            meanCBED if meanCBED is not None else np.mean(datacube.data, axis=(0, 1))
        )

        self.mask = mask if mask else np.ones_like(self.meanCBED)

        self.model = []
        self.model_param_inds = []

        self.nParams = 0
        self.use_jacobian = use_jacobian

        if hasattr(x0, "__iter__") and hasattr(y0, "__iter__"):
            # the initial position was specified with bounds
            try:
                self.global_xy0_lb = np.array([x0[1], y0[1]])
                self.global_xy0_ub = np.array([x0[2], y0[2]])
            except:
                self.global_xy0_lb = np.array([0.0, 0.0])
                self.global_xy0_ub = np.array([datacube.Q_Nx, datacube.Q_Ny])
            x0 = x0[0]
            y0 = y0[0]
        else:
            self.global_xy0_lb = np.array([0.0, 0.0])
            self.global_xy0_ub = np.array([datacube.Q_Nx, datacube.Q_Ny])

        # set up the global arguments
        self.global_args = {}

        self.global_args["global_x0"] = x0 if x0 else datacube.Q_Nx / 2.0
        self.global_args["global_y0"] = y0 if y0 else datacube.Q_Ny / 2.0

        xArray, yArray = np.mgrid[0 : datacube.Q_Nx, 0 : datacube.Q_Ny]
        self.global_args["xArray"] = xArray
        self.global_args["yArray"] = yArray

        self.global_args["global_r"] = np.hypot((xArray - x0) ** 2, (yArray - y0) ** 2)

        self.global_args["Q_Nx"] = datacube.Q_Nx
        self.global_args["Q_Ny"] = datacube.Q_Ny

        # for debugging: tracks all function evals
        self._track = True
        self._fevals = []
        self._xevals = []

    def add_model(self, model: WPFModelPrototype):
        self.model.append(model)

        # keep track of where each model's parameter list begins
        self.model_param_inds.append(self.nParams)
        self.nParams += len(model.params.keys())

        self._scrape_model_params()

    def add_model_list(self, model_list):
        for m in model_list:
            self.add_model(m)

    def generate_initial_pattern(self):

        # update parameters:
        self._scrape_model_params()

        # set the current active pattern to the mean CBED:
        self.current_pattern = self.meanCBED
        self.current_glob = self.global_args.copy()

        return self._pattern(self.x0).reshape(self.meanCBED.shape) + self.meanCBED

    def fit_to_mean_CBED(self, **fit_opts):

        # first make sure we have the latest parameters
        self._scrape_model_params()

        # set the current active pattern to the mean CBED:
        self.current_pattern = self.meanCBED
        self.current_glob = self.global_args.copy()

        self._fevals = []

        if self.hasJacobian & self.use_jacobian:
            opt = least_squares(
                self._pattern,
                self.x0,
                jac=self._jacobian,
                bounds=(self.lower_bound, self.upper_bound),
                **fit_opts
            )
        else:
            opt = least_squares(
                self._pattern,
                self.x0,
                bounds=(self.lower_bound, self.upper_bound),
                **fit_opts
            )

        self.mean_CBED_fit = opt

        return opt

    def accept_mean_CBED_fit(self):
        x = self.mean_CBED_fit.x
        self.global_args["global_x0"] = x[0]
        self.global_args["global_y0"] = x[1]

        self.global_args["global_r"] = np.hypot(
            (self.global_args["xArray"] - x[0]), (self.global_args["yArray"] - x[1])
        )

        for i, m in enumerate(self.model):
            ind = self.model_param_inds[i] + 2
            for j, k in enumerate(m.params.keys()):
                m.params[k].initial_value = x[ind + j]

    def _pattern(self, x):

        DP = np.zeros((self.datacube.Q_Nx, self.datacube.Q_Ny))

        self.current_glob["global_x0"] = x[0]
        self.current_glob["global_y0"] = x[1]
        self.current_glob["global_r"] = np.hypot(
            (self.current_glob["xArray"] - x[0]), (self.current_glob["yArray"] - x[1]),
        )

        for i, m in enumerate(self.model):
            ind = self.model_param_inds[i] + 2
            m.func(DP, *x[ind : ind + m.nParams].tolist(), **self.current_glob)

        DP = (DP - self.current_pattern) * self.mask

        if self._track:
            self._fevals.append(DP)
            self._xevals.append(x)

        return DP.ravel()

    def _jacobian(self, x):

        J = np.zeros(((self.datacube.Q_Nx * self.datacube.Q_Ny), self.nParams + 2))

        self.current_glob["global_x0"] = x[0]
        self.current_glob["global_y0"] = x[1]
        self.current_glob["global_r"] = np.hypot(
            (self.current_glob["xArray"] - x[0]), (self.current_glob["yArray"] - x[1]),
        )

        for i, m in enumerate(self.model):
            ind = self.model_param_inds[i] + 2
            m.jacobian(
                J, *x[ind : ind + m.nParams].tolist(), offset=ind, **self.current_glob
            )

        return J * self.mask.ravel()[:, np.newaxis]

    def _scrape_model_params(self):

        self.x0 = np.zeros((self.nParams + 2,))
        self.upper_bound = np.zeros_like(self.x0)
        self.lower_bound = np.zeros_like(self.x0)

        self.x0[0:2] = np.array(
            [self.global_args["global_x0"], self.global_args["global_y0"]]
        )
        self.upper_bound[0:2] = self.global_xy0_ub
        self.lower_bound[0:2] = self.global_xy0_lb

        for i, m in enumerate(self.model):
            ind = self.model_param_inds[i] + 2

            for j, v in enumerate(m.params.values()):
                self.x0[ind + j] = v.initial_value
                self.upper_bound[ind + j] = v.upper_bound
                self.lower_bound[ind + j] = v.lower_bound

        self.hasJacobian = all([m.hasJacobian for m in self.model])
