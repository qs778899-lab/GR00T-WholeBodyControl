"""Centralized Isaac-compatible wrench buffer writer with no Isaac import."""

from __future__ import annotations

from collections.abc import Sequence
import inspect

import torch

from .body_names import NamedSiteIndices, SiteIndexSpace
from .frames import (
    quaternion_rotate_inverse_wxyz,
    quaternion_rotate_inverse_wxyz_prevalidated,
)


class WrenchWriteGate:
    """Host-side ownership gate for globally optional permanent wrench writes."""

    def __init__(self) -> None:
        self._was_written = False

    @property
    def was_written(self) -> bool:
        return self._was_written

    def mark_written(self) -> None:
        self._was_written = True

    def consume_clear_on_disable(self) -> bool:
        should_clear = self._was_written
        self._was_written = False
        return should_clear

    def consume_clear_on_reset(self, *, globally_enabled: bool) -> bool:
        """Request a reset clear, consuming ownership only when globally off."""

        if not isinstance(globally_enabled, bool):
            raise TypeError("globally_enabled must be a bool")
        if globally_enabled:
            return True
        return self.consume_clear_on_disable()


class ArticulationWrenchAdapter:
    """Write persistent world-frame force-on-robot buffers by typed body names.

    Isaac Lab 2.3+ exposes ``permanent_wrench_composer``.  The deprecated
    articulation setter remains a narrow compatibility fallback so API-version
    handling is not scattered through events or commands.
    """

    def __init__(
        self,
        articulation: object,
        *,
        body_selection: NamedSiteIndices,
        num_envs: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        if not isinstance(body_selection, NamedSiteIndices):
            raise TypeError("body_selection must be a NamedSiteIndices")
        if body_selection.index_space is not SiteIndexSpace.ARTICULATION:
            raise ValueError("wrench application requires articulation-space indices")
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if not dtype.is_floating_point:
            raise TypeError("dtype must be floating point")
        self.articulation = articulation
        self.body_selection = body_selection
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        self._all_env_ids = torch.arange(
            num_envs,
            device=self.device,
            dtype=torch.long,
        )

    def _env_ids_tensor(
        self,
        env_ids: torch.Tensor | Sequence[int] | slice | None,
    ) -> torch.Tensor:
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            return self._all_env_ids
        if isinstance(env_ids, torch.Tensor):
            if env_ids.ndim != 1 or env_ids.dtype not in (torch.int32, torch.int64):
                raise TypeError("env_ids tensor must be one-dimensional int32/int64")
            ids = env_ids.to(device=self.device, dtype=torch.long)
        else:
            if isinstance(env_ids, str | bytes | slice):
                raise TypeError("env_ids must be an integer sequence or full slice")
            ids = torch.tensor(tuple(env_ids), device=self.device, dtype=torch.long)
        if ids.numel() and ((ids < 0).any() or (ids >= self.num_envs).any()):
            raise IndexError("env_ids contain an out-of-range environment index")
        return ids

    def set_world_forces(
        self,
        forces_on_robot_w: torch.Tensor,
        env_ids: torch.Tensor | Sequence[int] | slice | None = None,
        *,
        body_quaternions_wxyz: torch.Tensor,
        application_offsets_local: torch.Tensor | None = None,
    ) -> None:
        """Convert checked world forces to current body frames and persist them."""

        ids = self._env_ids_tensor(env_ids)
        expected_shape = (ids.numel(), len(self.body_selection.indices), 3)
        if not isinstance(forces_on_robot_w, torch.Tensor):
            raise TypeError("forces_on_robot_w must be a torch.Tensor")
        if tuple(forces_on_robot_w.shape) != expected_shape:
            raise ValueError(f"forces_on_robot_w must have shape {expected_shape}")
        if forces_on_robot_w.dtype != self.dtype:
            raise TypeError("forces_on_robot_w must use adapter dtype")
        if forces_on_robot_w.device != self.device:
            raise ValueError("forces_on_robot_w must use adapter device")
        if not torch.isfinite(forces_on_robot_w).all():
            raise ValueError("forces_on_robot_w must contain only finite values")
        expected_quaternion_shape = (*expected_shape[:-1], 4)
        if not isinstance(body_quaternions_wxyz, torch.Tensor):
            raise TypeError("body_quaternions_wxyz must be a torch.Tensor")
        if tuple(body_quaternions_wxyz.shape) != expected_quaternion_shape:
            raise ValueError(
                f"body_quaternions_wxyz must have shape {expected_quaternion_shape}"
            )
        if body_quaternions_wxyz.dtype != self.dtype:
            raise TypeError("body_quaternions_wxyz must use adapter dtype")
        if body_quaternions_wxyz.device != self.device:
            raise ValueError("body_quaternions_wxyz must use adapter device")
        if application_offsets_local is not None:
            if not isinstance(application_offsets_local, torch.Tensor):
                raise TypeError("application_offsets_local must be a torch.Tensor")
            if tuple(application_offsets_local.shape) != expected_shape:
                raise ValueError(f"application_offsets_local must have shape {expected_shape}")
            if application_offsets_local.dtype != self.dtype:
                raise TypeError("application_offsets_local must use adapter dtype")
            if application_offsets_local.device != self.device:
                raise ValueError("application_offsets_local must use adapter device")
            if not torch.isfinite(application_offsets_local).all():
                raise ValueError("application_offsets_local must contain only finite values")

        forces_body = quaternion_rotate_inverse_wxyz(
            body_quaternions_wxyz,
            forces_on_robot_w,
        )
        self._write_local_forces(
            forces_body,
            ids,
            application_offsets_local=application_offsets_local,
        )

    def set_world_forces_prevalidated(
        self,
        forces_on_robot_w: torch.Tensor,
        env_ids: torch.Tensor | Sequence[int] | slice | None = None,
        *,
        body_quaternions_wxyz: torch.Tensor,
        application_offsets_local: torch.Tensor | None = None,
    ) -> None:
        """Write lifecycle-validated world forces without CUDA host sync."""

        ids = self._env_ids_tensor(env_ids)
        forces_body = quaternion_rotate_inverse_wxyz_prevalidated(
            body_quaternions_wxyz,
            forces_on_robot_w,
        )
        self._write_local_forces(
            forces_body,
            ids,
            application_offsets_local=application_offsets_local,
        )

    def _write_local_forces(
        self,
        forces_body: torch.Tensor,
        ids: torch.Tensor,
        *,
        application_offsets_local: torch.Tensor | None,
    ) -> None:
        """Write body-local forces/offsets without stale composer link-pose use."""

        torques = torch.zeros_like(forces_body)
        body_ids = list(self.body_selection.indices)
        composer = getattr(self.articulation, "permanent_wrench_composer", None)
        composer_setter = getattr(composer, "set_forces_and_torques", None)
        if callable(composer_setter):
            composer_setter(
                forces=forces_body,
                torques=torques,
                positions=application_offsets_local,
                body_ids=body_ids,
                env_ids=ids,
                is_global=False,
            )
            return

        legacy_setter = getattr(self.articulation, "set_external_force_and_torque", None)
        if not callable(legacy_setter):
            raise AttributeError(
                "articulation exposes neither permanent_wrench_composer nor "
                "set_external_force_and_torque"
            )
        try:
            supports_global = "is_global" in inspect.signature(legacy_setter).parameters
        except (TypeError, ValueError):
            supports_global = False
        if not supports_global:
            raise RuntimeError(
                "legacy wrench API cannot declare body-local forces; Isaac Lab 2.3+ is required"
            )
        legacy_setter(
            forces=forces_body,
            torques=torques,
            positions=application_offsets_local,
            body_ids=body_ids,
            env_ids=ids,
            is_global=False,
        )

    def clear(
        self,
        env_ids: torch.Tensor | Sequence[int] | slice | None = None,
    ) -> None:
        """Overwrite selected persistent wrench rows with zero to prevent staleness."""

        ids = self._env_ids_tensor(env_ids)
        zeros = torch.zeros(
            ids.numel(),
            len(self.body_selection.indices),
            3,
            dtype=self.dtype,
            device=self.device,
        )
        self._write_local_forces(zeros, ids, application_offsets_local=None)
