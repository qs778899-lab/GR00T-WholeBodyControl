"""Tracker-neutral, frame-aligned tracking and compliance evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .schema import CartesianFrameSpec


def _ordered_names(names: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    if not isinstance(names, tuple):
        raise TypeError(f"{label} must be a tuple")
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(names) != len(set(names)):
        raise ValueError(f"{label} must be unique and ordered")
    return names


def _finite_float_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if shape is not None and tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _unit_quaternion_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    _finite_float_tensor(tensor, name=name, shape=shape)
    norms = torch.linalg.vector_norm(tensor, dim=-1)
    if (norms <= torch.finfo(tensor.dtype).eps).any():
        raise ValueError(f"{name} contains a zero quaternion")
    if not torch.allclose(norms, torch.ones_like(norms), rtol=0.0, atol=1.0e-5):
        raise ValueError(f"{name} must contain normalized wxyz quaternions")
    return tensor


def _integer_vector(tensor: torch.Tensor, *, name: str, length: int) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must use an integer dtype")
    if tuple(tensor.shape) != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    return tensor


@dataclass(frozen=True, slots=True)
class AlignedTrackingTrace:
    """One fixed-horizon rollout in explicit reference and Cartesian frames.

    ``reference_positions_local`` and ``actual_positions_local`` must already be
    expressed in their respective anchor-local frames.  This keeps coordinate
    conversion in a tracker adapter and leaves the metric implementation
    reusable.  Invalid samples after a fall remain finite but are excluded by
    ``valid``; no interpolation or nearest-frame matching is performed. Sample
    ``k`` is the state immediately before environment transition ``k``. If that
    transition terminates with a non-timeout fall, sample ``k`` remains valid,
    ``termination_sample == k``, and every later fixed-horizon sample is invalid
    even if the simulator auto-resets.
    """

    mode: str
    body_names: tuple[str, ...]
    site_names: tuple[str, ...]
    local_frame: CartesianFrameSpec
    sample_index: torch.Tensor
    episode_id: torch.Tensor
    motion_id: torch.Tensor
    reference_frame: torch.Tensor
    time_s: torch.Tensor
    valid: torch.Tensor
    reference_positions_w: torch.Tensor
    actual_positions_w: torch.Tensor
    reference_positions_local: torch.Tensor
    actual_positions_local: torch.Tensor
    reference_site_positions_w: torch.Tensor
    actual_site_positions_w: torch.Tensor
    reference_site_quaternions_wxyz: torch.Tensor
    actual_site_quaternions_wxyz: torch.Tensor
    force_on_robot_w: torch.Tensor
    enabled: torch.Tensor
    site_mask: torch.Tensor
    compliance_m_per_n: torch.Tensor
    fell: bool
    horizon_reached: bool
    termination_sample: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("stiff", "compliant"):
            raise ValueError("mode must be 'stiff' or 'compliant'")
        body_names = _ordered_names(self.body_names, label="body_names")
        site_names = _ordered_names(self.site_names, label="site_names")
        if not isinstance(self.local_frame, CartesianFrameSpec):
            raise TypeError("local_frame must be a CartesianFrameSpec")
        if type(self.fell) is not bool or type(self.horizon_reached) is not bool:
            raise TypeError("fell and horizon_reached must be bool")
        if self.fell == self.horizon_reached:
            raise ValueError("exactly one of fell and horizon_reached must be true")

        if not isinstance(self.valid, torch.Tensor) or self.valid.dtype is not torch.bool:
            raise TypeError("valid must be a boolean tensor")
        if self.valid.ndim != 1 or self.valid.numel() == 0:
            raise ValueError("valid must be a non-empty vector")
        samples = self.valid.numel()
        if not self.valid.any():
            raise ValueError("trace must contain at least one valid sample")
        for name, value in (
            ("sample_index", self.sample_index),
            ("episode_id", self.episode_id),
            ("motion_id", self.motion_id),
            ("reference_frame", self.reference_frame),
        ):
            _integer_vector(value, name=name, length=samples)
        expected_sample_index = torch.arange(
            samples,
            dtype=self.sample_index.dtype,
            device=self.sample_index.device,
        )
        if not torch.equal(self.sample_index, expected_sample_index):
            raise ValueError("sample_index must be contiguous from zero")
        _finite_float_tensor(self.time_s, name="time_s", shape=(samples,))
        if (self.time_s[1:] <= self.time_s[:-1]).any():
            raise ValueError("time_s must be strictly increasing")

        bodies = len(body_names)
        sites = len(site_names)
        for name, value in (
            ("reference_positions_w", self.reference_positions_w),
            ("actual_positions_w", self.actual_positions_w),
            ("reference_positions_local", self.reference_positions_local),
            ("actual_positions_local", self.actual_positions_local),
        ):
            _finite_float_tensor(value, name=name, shape=(samples, bodies, 3))
        for name, value in (
            ("reference_site_positions_w", self.reference_site_positions_w),
            ("actual_site_positions_w", self.actual_site_positions_w),
            ("force_on_robot_w", self.force_on_robot_w),
            ("compliance_m_per_n", self.compliance_m_per_n),
        ):
            _finite_float_tensor(value, name=name, shape=(samples, sites, 3))
        for name, value in (
            (
                "reference_site_quaternions_wxyz",
                self.reference_site_quaternions_wxyz,
            ),
            ("actual_site_quaternions_wxyz", self.actual_site_quaternions_wxyz),
        ):
            _unit_quaternion_tensor(value, name=name, shape=(samples, sites, 4))
        if (self.compliance_m_per_n < 0.0).any():
            raise ValueError("compliance_m_per_n must be non-negative")
        if not isinstance(self.enabled, torch.Tensor) or self.enabled.dtype is not torch.bool:
            raise TypeError("enabled must be a boolean tensor")
        if tuple(self.enabled.shape) != (samples,):
            raise ValueError(f"enabled must have shape ({samples},)")
        if not isinstance(self.site_mask, torch.Tensor) or self.site_mask.dtype is not torch.bool:
            raise TypeError("site_mask must be a boolean tensor")
        if tuple(self.site_mask.shape) != (samples, sites):
            raise ValueError(f"site_mask must have shape ({samples}, {sites})")
        tensor_fields = (
            "sample_index",
            "episode_id",
            "motion_id",
            "reference_frame",
            "time_s",
            "valid",
            "reference_positions_w",
            "actual_positions_w",
            "reference_positions_local",
            "actual_positions_local",
            "reference_site_positions_w",
            "actual_site_positions_w",
            "reference_site_quaternions_wxyz",
            "actual_site_quaternions_wxyz",
            "force_on_robot_w",
            "enabled",
            "site_mask",
            "compliance_m_per_n",
        )
        devices = {getattr(self, name).device for name in tensor_fields}
        if len(devices) != 1:
            raise ValueError("all trace tensors must use one device")
        spatial_fields = (
            "reference_positions_w",
            "actual_positions_w",
            "reference_positions_local",
            "actual_positions_local",
            "reference_site_positions_w",
            "actual_site_positions_w",
            "reference_site_quaternions_wxyz",
            "actual_site_quaternions_wxyz",
            "force_on_robot_w",
            "compliance_m_per_n",
        )
        spatial_dtypes = {getattr(self, name).dtype for name in spatial_fields}
        if len(spatial_dtypes) != 1:
            raise TypeError("all Cartesian/quaternion trace tensors must use one dtype")

        if self.fell:
            if type(self.termination_sample) is not int:
                raise ValueError("fallen traces require an integer termination_sample")
            if not 0 <= self.termination_sample < samples:
                raise ValueError("termination_sample is outside the trace")
            if not self.valid[: self.termination_sample + 1].all():
                raise ValueError("samples through termination must be a valid prefix")
            if self.valid[self.termination_sample + 1 :].any():
                raise ValueError("samples after termination must be invalid")
        else:
            if self.termination_sample is not None:
                raise ValueError("successful traces must not set termination_sample")
            if not self.valid.all():
                raise ValueError("successful traces require the complete horizon to be valid")


@dataclass(frozen=True, slots=True)
class TrackingComplianceMetrics:
    """Finite scalar and per-site summaries for one aligned rollout."""

    valid_frames: int
    success_rate: float
    fall_rate: float
    global_mpjpe_m: float
    local_mpjpe_m: float
    upper_endpoint_mpjpe_m: float
    exposed_upper_endpoint_mpjpe_m: float
    upper_endpoint_orientation_rmse_rad: float
    exposed_upper_endpoint_orientation_rmse_rad: float
    peak_force_n: float
    steady_force_mean_n: float
    per_site_exposed_frames: tuple[int, ...]
    per_site_endpoint_rmse_m: tuple[float, ...]
    per_site_endpoint_p95_m: tuple[float, ...]
    per_site_exposed_endpoint_rmse_m: tuple[float, ...]
    per_site_exposed_endpoint_p95_m: tuple[float, ...]
    per_site_unexposed_endpoint_rmse_m: tuple[float, ...]
    per_site_unexposed_endpoint_p95_m: tuple[float, ...]
    per_site_orientation_rmse_rad: tuple[float, ...]
    per_site_orientation_p95_rad: tuple[float, ...]
    per_site_exposed_orientation_rmse_rad: tuple[float, ...]
    per_site_exposed_orientation_p95_rad: tuple[float, ...]
    per_site_unexposed_orientation_rmse_rad: tuple[float, ...]
    per_site_unexposed_orientation_p95_rad: tuple[float, ...]
    per_site_peak_force_n: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PairedComplianceResponseMetrics:
    """Yield of the compliant policy relative to the matched stiff policy."""

    displacement_mean_m: float
    displacement_max_m: float
    displacement_along_force_mean_m: float
    per_site_displacement_mean_m: tuple[float, ...]
    per_site_displacement_max_m: tuple[float, ...]
    per_site_displacement_along_force_mean_m: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PairedEvaluationThresholds:
    """Explicit acceptance budgets prioritizing tracking preservation."""

    min_aligned_frames: int = 200
    min_exposed_frames_per_site: int = 20
    max_upper_endpoint_regression_m: float = 0.05
    max_upper_endpoint_orientation_regression_rad: float = 0.25
    max_global_mpjpe_regression_m: float = 0.03
    max_local_mpjpe_regression_m: float = 0.03
    min_paired_displacement_m: float = 1.0e-6
    min_compliant_success_rate: float = 1.0
    max_compliant_fall_rate: float = 0.0
    min_peak_force_n: float = 1.0
    max_peak_force_n: float = 30.0

    def __post_init__(self) -> None:
        if type(self.min_aligned_frames) is not int or self.min_aligned_frames <= 0:
            raise ValueError("min_aligned_frames must be a positive integer")
        if (
            type(self.min_exposed_frames_per_site) is not int
            or self.min_exposed_frames_per_site <= 0
        ):
            raise ValueError("min_exposed_frames_per_site must be a positive integer")
        for name in (
            "max_upper_endpoint_regression_m",
            "max_upper_endpoint_orientation_regression_rad",
            "max_global_mpjpe_regression_m",
            "max_local_mpjpe_regression_m",
            "min_paired_displacement_m",
            "min_compliant_success_rate",
            "max_compliant_fall_rate",
            "min_peak_force_n",
            "max_peak_force_n",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 <= self.min_compliant_success_rate <= 1.0:
            raise ValueError("min_compliant_success_rate must be within [0, 1]")
        if not 0.0 <= self.max_compliant_fall_rate <= 1.0:
            raise ValueError("max_compliant_fall_rate must be within [0, 1]")
        if self.max_peak_force_n < self.min_peak_force_n:
            raise ValueError("max_peak_force_n must be at least min_peak_force_n")


@dataclass(frozen=True, slots=True)
class PairedEvaluationResult:
    """Metrics and named acceptance checks for one stiff/compliant pair."""

    aligned_frames: int
    stiff: TrackingComplianceMetrics
    compliant: TrackingComplianceMetrics
    compliance_response: PairedComplianceResponseMetrics
    checks: tuple[tuple[str, bool], ...]

    @property
    def passed(self) -> bool:
        return all(passed for _, passed in self.checks)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    if selected.numel() == 0:
        return 0.0
    return float(selected.mean().item())


def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    if selected.numel() == 0:
        return 0.0
    return float(selected.max().item())


def _masked_rmse(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    if selected.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(torch.square(selected))).item())


def _masked_p95(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    if selected.numel() == 0:
        return 0.0
    return float(torch.quantile(selected, 0.95).item())


def _pulse_tail_mask(exposure: torch.Tensor, *, tail_fraction: float) -> torch.Tensor:
    """Select the final temporal window of every contiguous site exposure pulse."""

    if exposure.ndim != 2 or exposure.dtype is not torch.bool:
        raise ValueError("exposure must be a [time, site] boolean tensor")
    result = torch.zeros_like(exposure)
    for site in range(exposure.shape[1]):
        values = exposure[:, site].detach().cpu().tolist()
        start = None
        for index, active in enumerate((*values, False)):
            if active and start is None:
                start = index
            elif not active and start is not None:
                length = index - start
                tail = max(1, math.ceil(length * tail_fraction))
                result[index - tail : index, site] = True
                start = None
    return result


def summarize_tracking_trace(
    trace: AlignedTrackingTrace,
    *,
    force_exposure_threshold_n: float = 1.0e-4,
    steady_tail_fraction: float = 0.2,
    valid_mask: torch.Tensor | None = None,
) -> TrackingComplianceMetrics:
    """Summarize one trace, optionally on an exact common valid prefix."""

    if not math.isfinite(force_exposure_threshold_n) or force_exposure_threshold_n < 0.0:
        raise ValueError("force_exposure_threshold_n must be finite and non-negative")
    if not math.isfinite(steady_tail_fraction) or not 0.0 < steady_tail_fraction <= 1.0:
        raise ValueError("steady_tail_fraction must be within (0, 1]")

    if valid_mask is None:
        valid = trace.valid
    else:
        if not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype is not torch.bool:
            raise TypeError("valid_mask must be a boolean tensor")
        if tuple(valid_mask.shape) != tuple(trace.valid.shape):
            raise ValueError("valid_mask must match the trace horizon")
        if (valid_mask & ~trace.valid).any():
            raise ValueError("valid_mask must be a subset of trace.valid")
        if not valid_mask.any():
            raise ValueError("valid_mask must select at least one frame")
        false_seen = (~valid_mask).to(torch.int64).cumsum(dim=0) > 0
        if (valid_mask & false_seen).any():
            raise ValueError("valid_mask must select a contiguous prefix")
        valid = valid_mask
    global_error = torch.linalg.vector_norm(
        trace.actual_positions_w - trace.reference_positions_w,
        dim=-1,
    )
    local_error = torch.linalg.vector_norm(
        trace.actual_positions_local - trace.reference_positions_local,
        dim=-1,
    )
    endpoint_delta = trace.actual_site_positions_w - trace.reference_site_positions_w
    endpoint_error = torch.linalg.vector_norm(endpoint_delta, dim=-1)
    quaternion_dot = torch.abs(
        (
            trace.reference_site_quaternions_wxyz
            * trace.actual_site_quaternions_wxyz
        ).sum(dim=-1)
    ).clamp(max=1.0)
    orientation_error = 2.0 * torch.acos(quaternion_dot)
    force_norm = torch.linalg.vector_norm(trace.force_on_robot_w, dim=-1)
    compliance_active = (trace.compliance_m_per_n > 0.0).any(dim=-1)
    exposure = (
        valid.unsqueeze(-1)
        & trace.enabled.unsqueeze(-1)
        & trace.site_mask
        & compliance_active
        & (force_norm > force_exposure_threshold_n)
    )
    per_site_exposed_frames = tuple(int(value) for value in exposure.sum(dim=0).tolist())
    per_site_endpoint_rmse = tuple(
        _masked_rmse(endpoint_error[:, site], valid)
        for site in range(len(trace.site_names))
    )
    per_site_endpoint_p95 = tuple(
        _masked_p95(endpoint_error[:, site], valid)
        for site in range(len(trace.site_names))
    )
    per_site_exposed_endpoint_rmse = tuple(
        _masked_rmse(endpoint_error[:, site], exposure[:, site])
        for site in range(len(trace.site_names))
    )
    per_site_exposed_endpoint_p95 = tuple(
        _masked_p95(endpoint_error[:, site], exposure[:, site])
        for site in range(len(trace.site_names))
    )
    unexposed = valid.unsqueeze(-1) & ~exposure
    per_site_unexposed_endpoint_rmse = tuple(
        _masked_rmse(endpoint_error[:, site], unexposed[:, site])
        for site in range(len(trace.site_names))
    )
    per_site_unexposed_endpoint_p95 = tuple(
        _masked_p95(endpoint_error[:, site], unexposed[:, site])
        for site in range(len(trace.site_names))
    )
    per_site_orientation_rmse = tuple(
        _masked_rmse(orientation_error[:, site], valid)
        for site in range(len(trace.site_names))
    )
    per_site_orientation_p95 = tuple(
        _masked_p95(orientation_error[:, site], valid)
        for site in range(len(trace.site_names))
    )
    per_site_exposed_orientation_rmse = tuple(
        _masked_rmse(orientation_error[:, site], exposure[:, site])
        for site in range(len(trace.site_names))
    )
    per_site_exposed_orientation_p95 = tuple(
        _masked_p95(orientation_error[:, site], exposure[:, site])
        for site in range(len(trace.site_names))
    )
    per_site_unexposed_orientation_rmse = tuple(
        _masked_rmse(orientation_error[:, site], unexposed[:, site])
        for site in range(len(trace.site_names))
    )
    per_site_unexposed_orientation_p95 = tuple(
        _masked_p95(orientation_error[:, site], unexposed[:, site])
        for site in range(len(trace.site_names))
    )
    per_site_peak_force = tuple(
        _masked_max(force_norm[:, site], exposure[:, site])
        for site in range(len(trace.site_names))
    )
    steady_mask = _pulse_tail_mask(exposure, tail_fraction=steady_tail_fraction)

    return TrackingComplianceMetrics(
        valid_frames=int(valid.sum().item()),
        success_rate=1.0 if trace.horizon_reached else 0.0,
        fall_rate=1.0 if trace.fell else 0.0,
        global_mpjpe_m=_masked_mean(global_error, valid.unsqueeze(-1).expand_as(global_error)),
        local_mpjpe_m=_masked_mean(local_error, valid.unsqueeze(-1).expand_as(local_error)),
        upper_endpoint_mpjpe_m=_masked_mean(
            endpoint_error,
            valid.unsqueeze(-1).expand_as(endpoint_error),
        ),
        exposed_upper_endpoint_mpjpe_m=_masked_mean(endpoint_error, exposure),
        upper_endpoint_orientation_rmse_rad=_masked_rmse(
            orientation_error,
            valid.unsqueeze(-1).expand_as(orientation_error),
        ),
        exposed_upper_endpoint_orientation_rmse_rad=_masked_rmse(
            orientation_error,
            exposure,
        ),
        peak_force_n=_masked_max(force_norm, exposure),
        steady_force_mean_n=_masked_mean(force_norm, steady_mask),
        per_site_exposed_frames=per_site_exposed_frames,
        per_site_endpoint_rmse_m=per_site_endpoint_rmse,
        per_site_endpoint_p95_m=per_site_endpoint_p95,
        per_site_exposed_endpoint_rmse_m=per_site_exposed_endpoint_rmse,
        per_site_exposed_endpoint_p95_m=per_site_exposed_endpoint_p95,
        per_site_unexposed_endpoint_rmse_m=per_site_unexposed_endpoint_rmse,
        per_site_unexposed_endpoint_p95_m=per_site_unexposed_endpoint_p95,
        per_site_orientation_rmse_rad=per_site_orientation_rmse,
        per_site_orientation_p95_rad=per_site_orientation_p95,
        per_site_exposed_orientation_rmse_rad=per_site_exposed_orientation_rmse,
        per_site_exposed_orientation_p95_rad=per_site_exposed_orientation_p95,
        per_site_unexposed_orientation_rmse_rad=per_site_unexposed_orientation_rmse,
        per_site_unexposed_orientation_p95_rad=per_site_unexposed_orientation_p95,
        per_site_peak_force_n=per_site_peak_force,
    )


def compare_aligned_tracking_traces(
    stiff: AlignedTrackingTrace,
    compliant: AlignedTrackingTrace,
    *,
    thresholds: PairedEvaluationThresholds = PairedEvaluationThresholds(),
    alignment_atol: float = 1.0e-6,
) -> PairedEvaluationResult:
    """Compare an exactly keyed pair and return explicit acceptance checks."""

    if stiff.mode != "stiff" or compliant.mode != "compliant":
        raise ValueError("compare requires stiff then compliant traces")
    if stiff.body_names != compliant.body_names or stiff.site_names != compliant.site_names:
        raise ValueError("paired traces must use the same ordered body/site contracts")
    if stiff.local_frame != compliant.local_frame:
        raise ValueError("paired traces must use the same structured local frame")
    if stiff.valid.shape != compliant.valid.shape:
        raise ValueError("paired traces must use one fixed horizon")
    if not math.isfinite(alignment_atol) or alignment_atol < 0.0:
        raise ValueError("alignment_atol must be finite and non-negative")

    common = stiff.valid & compliant.valid
    aligned_frames = int(common.sum().item())
    if aligned_frames == 0:
        raise ValueError("paired traces have no commonly valid frames")
    for name in ("sample_index", "episode_id", "motion_id", "reference_frame"):
        left = getattr(stiff, name)[common]
        right = getattr(compliant, name)[common]
        if not torch.equal(left, right):
            raise ValueError(f"paired {name} values are not exactly aligned")
    for name in (
        "time_s",
        "reference_positions_w",
        "reference_positions_local",
        "reference_site_positions_w",
        "force_on_robot_w",
        "compliance_m_per_n",
        "reference_site_quaternions_wxyz",
    ):
        left = getattr(stiff, name)[common]
        right = getattr(compliant, name)[common]
        if not torch.allclose(left, right, rtol=0.0, atol=alignment_atol):
            raise ValueError(f"paired {name} values are not frame aligned")
    if not torch.equal(stiff.enabled[common], compliant.enabled[common]):
        raise ValueError("paired enabled gates are not frame aligned")
    if not torch.equal(stiff.site_mask[common], compliant.site_mask[common]):
        raise ValueError("paired site masks are not frame aligned")

    stiff_metrics = summarize_tracking_trace(stiff, valid_mask=common)
    compliant_metrics = summarize_tracking_trace(compliant, valid_mask=common)
    force_norm = torch.linalg.vector_norm(compliant.force_on_robot_w, dim=-1)
    paired_exposure = (
        common.unsqueeze(-1)
        & compliant.enabled.unsqueeze(-1)
        & compliant.site_mask
        & (compliant.compliance_m_per_n > 0.0).any(dim=-1)
        & (force_norm > 1.0e-4)
    )
    yielding_delta = compliant.actual_site_positions_w - stiff.actual_site_positions_w
    yielding_displacement = torch.linalg.vector_norm(yielding_delta, dim=-1)
    safe_force_direction = compliant.force_on_robot_w / force_norm.clamp_min(
        torch.finfo(force_norm.dtype).eps
    ).unsqueeze(-1)
    yielding_along_force = (yielding_delta * safe_force_direction).sum(dim=-1)
    compliance_response = PairedComplianceResponseMetrics(
        displacement_mean_m=_masked_mean(yielding_displacement, paired_exposure),
        displacement_max_m=_masked_max(yielding_displacement, paired_exposure),
        displacement_along_force_mean_m=_masked_mean(
            yielding_along_force,
            paired_exposure,
        ),
        per_site_displacement_mean_m=tuple(
            _masked_mean(yielding_displacement[:, site], paired_exposure[:, site])
            for site in range(len(compliant.site_names))
        ),
        per_site_displacement_max_m=tuple(
            _masked_max(yielding_displacement[:, site], paired_exposure[:, site])
            for site in range(len(compliant.site_names))
        ),
        per_site_displacement_along_force_mean_m=tuple(
            _masked_mean(yielding_along_force[:, site], paired_exposure[:, site])
            for site in range(len(compliant.site_names))
        ),
    )
    aggregate_checks = (
        ("aligned_frames", aligned_frames >= thresholds.min_aligned_frames),
        (
            "per_site_force_exposure",
            all(
                count >= thresholds.min_exposed_frames_per_site
                for count in compliant_metrics.per_site_exposed_frames
            ),
        ),
        (
            "upper_endpoint_tracking_budget",
            compliant_metrics.upper_endpoint_mpjpe_m
            <= stiff_metrics.upper_endpoint_mpjpe_m
            + thresholds.max_upper_endpoint_regression_m,
        ),
        (
            "upper_endpoint_orientation_budget",
            compliant_metrics.upper_endpoint_orientation_rmse_rad
            <= stiff_metrics.upper_endpoint_orientation_rmse_rad
            + thresholds.max_upper_endpoint_orientation_regression_rad,
        ),
    )
    per_site_checks: list[tuple[str, bool]] = []
    for site_index, site_name in enumerate(compliant.site_names):
        per_site_checks.extend(
            (
                (
                    f"site/{site_name}/position_rmse_budget",
                    compliant_metrics.per_site_endpoint_rmse_m[site_index]
                    <= stiff_metrics.per_site_endpoint_rmse_m[site_index]
                    + thresholds.max_upper_endpoint_regression_m,
                ),
                (
                    f"site/{site_name}/position_p95_budget",
                    compliant_metrics.per_site_endpoint_p95_m[site_index]
                    <= stiff_metrics.per_site_endpoint_p95_m[site_index]
                    + thresholds.max_upper_endpoint_regression_m,
                ),
                (
                    f"site/{site_name}/orientation_rmse_budget",
                    compliant_metrics.per_site_orientation_rmse_rad[site_index]
                    <= stiff_metrics.per_site_orientation_rmse_rad[site_index]
                    + thresholds.max_upper_endpoint_orientation_regression_rad,
                ),
                (
                    f"site/{site_name}/orientation_p95_budget",
                    compliant_metrics.per_site_orientation_p95_rad[site_index]
                    <= stiff_metrics.per_site_orientation_p95_rad[site_index]
                    + thresholds.max_upper_endpoint_orientation_regression_rad,
                ),
            )
        )
    outcome_checks = (
        (
            "global_tracking_budget",
            compliant_metrics.global_mpjpe_m
            <= stiff_metrics.global_mpjpe_m
            + thresholds.max_global_mpjpe_regression_m,
        ),
        (
            "local_tracking_budget",
            compliant_metrics.local_mpjpe_m
            <= stiff_metrics.local_mpjpe_m
            + thresholds.max_local_mpjpe_regression_m,
        ),
        (
            "paired_displacement_activation",
            compliance_response.displacement_mean_m
            >= thresholds.min_paired_displacement_m,
        ),
        (
            "success_rate",
            compliant_metrics.success_rate >= thresholds.min_compliant_success_rate,
        ),
        (
            "fall_rate",
            compliant_metrics.fall_rate <= thresholds.max_compliant_fall_rate,
        ),
        (
            "force_minimum",
            compliant_metrics.peak_force_n >= thresholds.min_peak_force_n,
        ),
        (
            "force_cap",
            compliant_metrics.peak_force_n <= thresholds.max_peak_force_n,
        ),
    )
    checks = aggregate_checks + tuple(per_site_checks) + outcome_checks
    return PairedEvaluationResult(
        aligned_frames=aligned_frames,
        stiff=stiff_metrics,
        compliant=compliant_metrics,
        compliance_response=compliance_response,
        checks=checks,
    )
