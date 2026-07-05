"""GPyTorch reconstruction with a center-decay mean component."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

try:
    import gpytorch
    import torch
    from gpytorch.utils.warnings import NumericalWarning
except ImportError as exc:  # pragma: no cover - exercised only without optional deps
    raise ImportError(
        "GPyTorch reconstruction requires torch and gpytorch. "
        "Install the repository environment or run: pip install torch gpytorch"
    ) from exc


@dataclass(frozen=True)
class MaternComponentSpec:
    """Manuscript Matern component definition."""

    label: str
    nu: float
    length_scale_um: float
    output_scale: float


MANUSCRIPT_MATERN_COMPONENTS: tuple[MaternComponentSpec, ...] = (
    MaternComponentSpec(
        label="Matern 1",
        nu=1.5,
        length_scale_um=17.56,
        output_scale=628.01,
    ),
    MaternComponentSpec(
        label="Matern 2",
        nu=0.5,
        length_scale_um=9.28,
        output_scale=457.97,
    ),
    MaternComponentSpec(
        label="Matern 3",
        nu=2.5,
        length_scale_um=42.31,
        output_scale=552.50,
    ),
)

RBF_LENGTH_UM = 20.0
DEFAULT_LENGTH_SCALE_FACTOR_BOUNDS = (0.75, 1.25)


def center_decay_mean(
    coordinates: np.ndarray,
    centers: np.ndarray,
    baseline: float,
    amplitude: float,
    length_scale_um: float,
) -> np.ndarray:
    """Evaluate a radial exponential center-decay mean function."""

    coords = np.asarray(coordinates, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if coords.ndim != 2:
        raise ValueError("coordinates must be a 2D array.")
    if length_scale_um <= 0:
        raise ValueError("length_scale_um must be positive.")
    if centers.size == 0:
        return np.full(coords.shape[0], baseline, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != coords.shape[1]:
        raise ValueError("centers must have shape (n_centers, n_dimensions).")
    distances = np.linalg.norm(coords[:, None, :] - centers[None, :, :], axis=2)
    nearest = distances.min(axis=1)
    return baseline + amplitude * np.exp(-nearest / float(length_scale_um))


def _torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    lookup = {
        "float32": torch.float32,
        "single": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
    }
    try:
        return lookup[dtype.lower()]
    except KeyError as exc:
        raise ValueError("dtype must be 'float32' or 'float64'.") from exc


def _scaled_bounds(value: float, factor_bounds: tuple[float, float] | None) -> tuple[float, float] | None:
    if factor_bounds is None:
        return None
    lower, upper = factor_bounds
    if lower <= 0 or upper <= lower:
        raise ValueError("length_scale_factor_bounds must be positive and increasing.")
    return float(value * lower), float(value * upper)


def _lengthscale_constraint(
    value: float,
    factor_bounds: tuple[float, float] | None,
) -> gpytorch.constraints.Interval | None:
    bounds = _scaled_bounds(value, factor_bounds)
    if bounds is None:
        return None
    return gpytorch.constraints.Interval(*bounds)


def build_composite_matern_kernel(
    *,
    fixed: bool = True,
    length_scale_factor_bounds: tuple[float, float] | None = DEFAULT_LENGTH_SCALE_FACTOR_BOUNDS,
    dtype: str | torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> gpytorch.kernels.Kernel:
    """Build the three-component Matern covariance used for reconstruction."""

    torch_dtype = _torch_dtype(dtype)
    torch_device = torch.device(device)
    modules: list[gpytorch.kernels.Kernel] = []
    for component in MANUSCRIPT_MATERN_COMPONENTS:
        base = gpytorch.kernels.MaternKernel(
            nu=component.nu,
            lengthscale_constraint=_lengthscale_constraint(
                component.length_scale_um,
                length_scale_factor_bounds,
            ),
        )
        kernel = gpytorch.kernels.ScaleKernel(base).to(
            dtype=torch_dtype,
            device=torch_device,
        )
        kernel.base_kernel.lengthscale = float(component.length_scale_um)
        kernel.outputscale = float(component.output_scale)
        modules.append(kernel)

    covar = modules[0]
    for component in modules[1:]:
        covar = covar + component
    if fixed:
        for parameter in covar.parameters():
            parameter.requires_grad_(False)
    return covar


def build_rbf_kernel(
    *,
    length_scale_um: float = RBF_LENGTH_UM,
    output_scale: float = 1.0,
    fixed: bool = True,
    length_scale_factor_bounds: tuple[float, float] | None = DEFAULT_LENGTH_SCALE_FACTOR_BOUNDS,
    dtype: str | torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> gpytorch.kernels.Kernel:
    """Build a single RBF covariance module."""

    if length_scale_um <= 0:
        raise ValueError("length_scale_um must be positive.")
    if output_scale <= 0:
        raise ValueError("output_scale must be positive.")
    torch_dtype = _torch_dtype(dtype)
    torch_device = torch.device(device)
    kernel = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(
            lengthscale_constraint=_lengthscale_constraint(
                length_scale_um,
                length_scale_factor_bounds,
            )
        )
    ).to(
        dtype=torch_dtype,
        device=torch_device,
    )
    kernel.base_kernel.lengthscale = float(length_scale_um)
    kernel.outputscale = float(output_scale)
    if fixed:
        for parameter in kernel.parameters():
            parameter.requires_grad_(False)
    return kernel


def build_covariance_kernel(
    kernel: str,
    *,
    rbf_length_um: float = RBF_LENGTH_UM,
    rbf_output_scale: float = 1.0,
    fixed: bool = True,
    length_scale_factor_bounds: tuple[float, float] | None = DEFAULT_LENGTH_SCALE_FACTOR_BOUNDS,
    dtype: str | torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> gpytorch.kernels.Kernel:
    """Build a covariance module by name."""

    if kernel == "composite_matern":
        return build_composite_matern_kernel(
            fixed=fixed,
            length_scale_factor_bounds=length_scale_factor_bounds,
            dtype=dtype,
            device=device,
        )
    if kernel == "rbf":
        return build_rbf_kernel(
            length_scale_um=rbf_length_um,
            output_scale=rbf_output_scale,
            fixed=fixed,
            length_scale_factor_bounds=length_scale_factor_bounds,
            dtype=dtype,
            device=device,
        )
    raise ValueError("kernel must be 'composite_matern' or 'rbf'.")


class _ExactResidualGP(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        kernel: str,
        rbf_length_um: float,
        rbf_output_scale: float,
        fixed_kernel: bool,
        length_scale_factor_bounds: tuple[float, float] | None,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = build_covariance_kernel(
            kernel,
            rbf_length_um=rbf_length_um,
            rbf_output_scale=rbf_output_scale,
            fixed=fixed_kernel,
            length_scale_factor_bounds=length_scale_factor_bounds,
            dtype=train_x.dtype,
            device=train_x.device,
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean = self.mean_module(x)
        covariance = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covariance)


@dataclass
class CompositeMaternGPR:
    """Exact GPyTorch GP fitted to residuals from a center-decay mean."""

    centers: np.ndarray | None = None
    baseline: float | None = None
    amplitude: float | None = None
    decay_length_um: float = 30.0
    kernel: str = "composite_matern"
    rbf_length_um: float = RBF_LENGTH_UM
    rbf_output_scale: float = 1.0
    length_scale_factor_bounds: tuple[float, float] | None = DEFAULT_LENGTH_SCALE_FACTOR_BOUNDS
    noise_variance: float = 1e-4
    noise_lower_bound: float = 1e-6
    normalize_y: bool = True
    optimize_kernel: bool = False
    optimize_output_scale: bool = False
    optimize_noise: bool = False
    training_iterations: int = 10000
    learning_rate: float = 0.05
    random_state: int = 0
    value_transform: str = "log"
    dtype: str | torch.dtype = "float64"
    device: str | torch.device = "cpu"
    jitter: float = 1e-6

    def fit(self, coordinates: np.ndarray, values: np.ndarray) -> "CompositeMaternGPR":
        coords = np.asarray(coordinates, dtype=float)
        y = np.asarray(values, dtype=float).reshape(-1)
        if coords.ndim != 2:
            raise ValueError("coordinates must be a 2D array.")
        if len(coords) != len(y):
            raise ValueError("coordinates and values must have the same length.")
        if np.any(~np.isfinite(coords)) or np.any(~np.isfinite(y)):
            raise ValueError("coordinates and values must be finite.")
        if self.noise_variance <= 0:
            raise ValueError("noise_variance must be positive.")
        if self.noise_lower_bound <= 0:
            raise ValueError("noise_lower_bound must be positive.")
        if self.noise_variance < self.noise_lower_bound:
            raise ValueError("noise_variance must be greater than or equal to noise_lower_bound.")
        if self.optimize_output_scale and not self.optimize_kernel:
            raise ValueError("optimize_output_scale requires optimize_kernel=True.")
        _scaled_bounds(1.0, self.length_scale_factor_bounds)

        baseline = float(np.median(y)) if self.baseline is None else float(self.baseline)
        amplitude = (
            float(max(np.percentile(y, 95) - baseline, 0.0))
            if self.amplitude is None
            else float(self.amplitude)
        )
        centers = (
            np.empty((0, coords.shape[1]))
            if self.centers is None
            else np.asarray(self.centers, dtype=float)
        )
        train_mean = center_decay_mean(
            coords,
            centers,
            baseline,
            amplitude,
            self.decay_length_um,
        )

        if self.value_transform not in {"identity", "log"}:
            raise ValueError("value_transform must be 'identity' or 'log'.")

        if self.value_transform == "log":
            if np.any(y <= 0):
                raise ValueError("value_transform='log' requires strictly positive values.")
            if np.any(train_mean <= 0):
                raise ValueError("value_transform='log' requires a strictly positive mean.")
            residual = np.log(y) - np.log(train_mean)
        else:
            residual = y - train_mean

        if self.normalize_y:
            residual_mean = float(np.mean(residual))
            residual_std = float(np.std(residual))
            if residual_std <= 0:
                residual_std = 1.0
            train_target = (residual - residual_mean) / residual_std
        else:
            residual_mean = 0.0
            residual_std = 1.0
            train_target = residual

        torch.manual_seed(int(self.random_state))
        torch_dtype = _torch_dtype(self.dtype)
        torch_device = torch.device(self.device)
        train_x = torch.as_tensor(coords, dtype=torch_dtype, device=torch_device)
        train_y = torch.as_tensor(train_target, dtype=torch_dtype, device=torch_device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(float(self.noise_lower_bound))
        ).to(dtype=torch_dtype, device=torch_device)
        likelihood.noise = float(self.noise_variance)
        if not self.optimize_noise:
            for parameter in likelihood.parameters():
                parameter.requires_grad_(False)

        model = _ExactResidualGP(
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            kernel=self.kernel,
            rbf_length_um=self.rbf_length_um,
            rbf_output_scale=self.rbf_output_scale,
            fixed_kernel=not self.optimize_kernel,
            length_scale_factor_bounds=self.length_scale_factor_bounds,
        ).to(dtype=torch_dtype, device=torch_device)

        if self.optimize_kernel and not self.optimize_output_scale:
            for module in model.covar_module.modules():
                if isinstance(module, gpytorch.kernels.ScaleKernel):
                    module.raw_outputscale.requires_grad_(False)

        if (self.optimize_kernel or self.optimize_noise) and self.training_iterations > 0:
            model.train()
            likelihood.train()
            trainable_parameters = [
                parameter
                for parameter in list(model.parameters()) + list(likelihood.parameters())
                if parameter.requires_grad
            ]
            optimizer = torch.optim.Adam(trainable_parameters, lr=float(self.learning_rate))
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            with gpytorch.settings.cholesky_jitter(float(self.jitter)):
                for _ in range(int(self.training_iterations)):
                    optimizer.zero_grad()
                    output = model(train_x)
                    loss = -mll(output, train_y)
                    loss.backward()
                    optimizer.step()

        model.eval()
        likelihood.eval()
        self._coordinates = coords
        self._values = y
        self._baseline = baseline
        self._amplitude = amplitude
        self._centers = centers
        self._residual_mean = residual_mean
        self._residual_std = residual_std
        self._model = model
        self._likelihood = likelihood
        self.kernel_ = model.covar_module
        return self

    def predict(self, coordinates: np.ndarray, return_std: bool = True):
        if not hasattr(self, "_model"):
            raise RuntimeError("The GPR model must be fitted before prediction.")
        coords = np.asarray(coordinates, dtype=float)
        if coords.ndim != 2:
            raise ValueError("coordinates must be a 2D array.")

        prior_mean = center_decay_mean(
            coords,
            self._centers,
            self._baseline,
            self._amplitude,
            self.decay_length_um,
        )
        torch_dtype = _torch_dtype(self.dtype)
        torch_device = torch.device(self.device)
        test_x = torch.as_tensor(coords, dtype=torch_dtype, device=torch_device)

        with (
            torch.no_grad(),
            gpytorch.settings.cholesky_jitter(float(self.jitter)),
            gpytorch.settings.min_variance(float(self.jitter)),
        ):
            posterior = self._model(test_x)
            residual = posterior.mean.detach().cpu().numpy()
            residual = residual * self._residual_std + self._residual_mean
            if return_std:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Negative variance values detected.*",
                        category=NumericalWarning,
                    )
                    variance = posterior.variance.detach().cpu().numpy()
                std = np.sqrt(np.maximum(variance, 0.0)) * self._residual_std

        if self.value_transform == "log":
            if np.any(prior_mean <= 0):
                raise ValueError("value_transform='log' requires a strictly positive mean.")
            prediction = np.exp(np.log(prior_mean) + residual)
            if return_std:
                return prediction, prediction * std
            return prediction

        prediction = prior_mean + residual
        if return_std:
            return prediction, std
        return prediction
