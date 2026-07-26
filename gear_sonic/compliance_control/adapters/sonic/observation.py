"""Non-mutating SONIC observation construction for compliant sparse targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...core import apply_hindsight_target, apply_hindsight_target_prevalidated
from .body_names import NamedSiteIndices, SiteIndexSpace
from .frames import (
    quaternion_rotate_wxyz,
    quaternion_rotate_wxyz_prevalidated,
    world_positions_to_frame,
    world_positions_to_frame_prevalidated,
    world_vectors_to_frame,
    world_vectors_to_frame_prevalidated,
)
from .state import SonicComplianceCommandState


@dataclass(frozen=True, slots=True)
class SonicComplianceTargets:
    """Named intermediate tensors produced by the Phase-2 observation boundary.

    ``force_on_robot_common`` has shape ``[env, future, site, xyz]``. The
    current measured/applied force is repeated over the future target horizon;
    it is not a prediction of future pulse phase.
    """

    reference_target_common: torch.Tensor
    damped_target_common: torch.Tensor
    nominal_target_common: torch.Tensor
    observed_target_common: torch.Tensor
    force_on_robot_common: torch.Tensor
    compliance: torch.Tensor
    enabled: torch.Tensor
    site_mask: torch.Tensor
    damper_used: bool


def select_reference_sites(
    reference_positions_w: torch.Tensor,
    selection: NamedSiteIndices,
) -> torch.Tensor:
    """Select `[env, future, site, xyz]` using only reference-space indices."""

    if not isinstance(selection, NamedSiteIndices):
        raise TypeError("selection must be a NamedSiteIndices")
    if selection.index_space is not SiteIndexSpace.REFERENCE:
        raise ValueError("reference positions require reference-space indices")
    if not isinstance(reference_positions_w, torch.Tensor):
        raise TypeError("reference_positions_w must be a torch.Tensor")
    if reference_positions_w.ndim != 4 or reference_positions_w.shape[-1] != 3:
        raise ValueError("reference_positions_w must have shape [env, future, body, xyz]")
    if selection.indices and max(selection.indices) >= reference_positions_w.shape[-2]:
        raise IndexError("reference-space site index exceeds reference body dimension")
    indices = torch.tensor(selection.indices, dtype=torch.long, device=reference_positions_w.device)
    return reference_positions_w.index_select(-2, indices)


def select_articulation_sites(
    articulation_positions_w: torch.Tensor,
    selection: NamedSiteIndices,
) -> torch.Tensor:
    """Select `[env, site, xyz]` using only articulation-space indices."""

    if not isinstance(selection, NamedSiteIndices):
        raise TypeError("selection must be a NamedSiteIndices")
    if selection.index_space is not SiteIndexSpace.ARTICULATION:
        raise ValueError("articulation positions require articulation-space indices")
    if not isinstance(articulation_positions_w, torch.Tensor):
        raise TypeError("articulation_positions_w must be a torch.Tensor")
    if articulation_positions_w.ndim != 3 or articulation_positions_w.shape[-1] != 3:
        raise ValueError("articulation_positions_w must have shape [env, body, xyz]")
    if selection.indices and max(selection.indices) >= articulation_positions_w.shape[-2]:
        raise IndexError("articulation-space site index exceeds articulation body dimension")
    indices = torch.tensor(
        selection.indices,
        dtype=torch.long,
        device=articulation_positions_w.device,
    )
    return articulation_positions_w.index_select(-2, indices)


def select_reference_site_quaternions(
    reference_quaternions_wxyz: torch.Tensor,
    selection: NamedSiteIndices,
) -> torch.Tensor:
    """Select `[env, future, site, wxyz]` using reference-space indices."""

    if selection.index_space is not SiteIndexSpace.REFERENCE:
        raise ValueError("reference quaternions require reference-space indices")
    if reference_quaternions_wxyz.ndim != 4 or reference_quaternions_wxyz.shape[-1] != 4:
        raise ValueError(
            "reference_quaternions_wxyz must have shape [env, future, body, wxyz]"
        )
    if selection.indices and max(selection.indices) >= reference_quaternions_wxyz.shape[-2]:
        raise IndexError("reference-space site index exceeds reference body dimension")
    indices = torch.tensor(
        selection.indices,
        dtype=torch.long,
        device=reference_quaternions_wxyz.device,
    )
    return reference_quaternions_wxyz.index_select(-2, indices)


def select_articulation_site_quaternions(
    articulation_quaternions_wxyz: torch.Tensor,
    selection: NamedSiteIndices,
) -> torch.Tensor:
    """Select `[env, site, wxyz]` using articulation-space indices."""

    if selection.index_space is not SiteIndexSpace.ARTICULATION:
        raise ValueError("articulation quaternions require articulation-space indices")
    if articulation_quaternions_wxyz.ndim != 3 or articulation_quaternions_wxyz.shape[-1] != 4:
        raise ValueError(
            "articulation_quaternions_wxyz must have shape [env, body, wxyz]"
        )
    if selection.indices and max(selection.indices) >= articulation_quaternions_wxyz.shape[-2]:
        raise IndexError("articulation-space site index exceeds articulation body dimension")
    indices = torch.tensor(
        selection.indices,
        dtype=torch.long,
        device=articulation_quaternions_wxyz.device,
    )
    return articulation_quaternions_wxyz.index_select(-2, indices)


def _build_sonic_compliance_targets(
    *,
    reference_positions_w: torch.Tensor,
    reference_quaternions_wxyz: torch.Tensor,
    articulation_positions_w: torch.Tensor,
    articulation_quaternions_wxyz: torch.Tensor,
    anchor_position_w: torch.Tensor | None,
    anchor_quaternion_wxyz: torch.Tensor | None,
    state: SonicComplianceCommandState,
    use_target_damper: bool = False,
    prevalidated: bool,
) -> SonicComplianceTargets:
    rotate = (
        quaternion_rotate_wxyz_prevalidated
        if prevalidated
        else quaternion_rotate_wxyz
    )
    transform_positions = (
        world_positions_to_frame_prevalidated
        if prevalidated
        else world_positions_to_frame
    )
    transform_vectors = (
        world_vectors_to_frame_prevalidated
        if prevalidated
        else world_vectors_to_frame
    )

    reference_w = select_reference_sites(reference_positions_w, state.sites.reference)
    current_eef_w = select_articulation_sites(
        articulation_positions_w,
        state.sites.articulation,
    )
    reference_site_quaternion = select_reference_site_quaternions(
        reference_quaternions_wxyz,
        state.sites.reference,
    )
    articulation_site_quaternion = select_articulation_site_quaternions(
        articulation_quaternions_wxyz,
        state.sites.articulation,
    )
    offsets = state.site_offsets_local
    reference_w = reference_w + rotate(
        reference_site_quaternion,
        offsets.view(1, 1, state.sites.spec.num_sites, 3).expand_as(reference_w),
    )
    current_eef_w = current_eef_w + rotate(
        articulation_site_quaternion,
        offsets.view(1, state.sites.spec.num_sites, 3).expand_as(current_eef_w),
    )
    reference_common = transform_positions(
        reference_w,
        frame=state.sites.spec.common_frame,
        anchor_position_w=anchor_position_w,
        anchor_quaternion_wxyz=anchor_quaternion_wxyz,
    )
    current_eef_common = transform_positions(
        current_eef_w,
        frame=state.sites.spec.common_frame,
        anchor_position_w=anchor_position_w,
        anchor_quaternion_wxyz=anchor_quaternion_wxyz,
    ).unsqueeze(1).expand(
        state.num_envs,
        state.num_future_frames,
        state.sites.spec.num_sites,
        3,
    )
    force_common_static = transform_vectors(
        state.force_on_robot_w,
        frame=state.sites.spec.common_frame,
        anchor_quaternion_wxyz=anchor_quaternion_wxyz,
    )
    force_common = force_common_static.unsqueeze(1).expand(
        state.num_envs,
        state.num_future_frames,
        state.sites.spec.num_sites,
        3,
    )

    enabled = state.enabled
    site_mask = state.site_mask
    compliance = state.compliance
    if use_target_damper:
        if not state.damper_initialized:
            raise RuntimeError("target damper must be reset before observation construction")
        damped_target = state.damped_target_common
        if damped_target.shape != reference_common.shape:
            raise ValueError("cached target-damper shape does not match reference target")
        damper_gate = (
            enabled[:, None, None]
            & site_mask[:, None, :]
            & (compliance > 0.0).any(dim=-1)[:, None, :]
        ).expand(
            state.num_envs,
            state.num_future_frames,
            state.sites.spec.num_sites,
        )
        nominal_target = torch.where(
            damper_gate.unsqueeze(-1),
            damped_target,
            reference_common,
        )
    else:
        damped_target = (
            state.damped_target_common
            if state.damper_initialized
            else current_eef_common.clone()
        )
        nominal_target = reference_common.clone()

    apply_target = (
        apply_hindsight_target_prevalidated
        if prevalidated
        else apply_hindsight_target
    )
    observed_target = apply_target(
        nominal_target,
        force_common,
        compliance,
        spec=state.sites.spec,
        enabled=enabled,
        site_mask=site_mask,
    )

    return SonicComplianceTargets(
        reference_target_common=reference_common,
        damped_target_common=damped_target,
        nominal_target_common=nominal_target,
        observed_target_common=observed_target,
        force_on_robot_common=force_common,
        compliance=compliance,
        enabled=enabled,
        site_mask=site_mask,
        damper_used=use_target_damper,
    )


def _validate_builder_inputs(
    *,
    reference_positions_w: torch.Tensor,
    articulation_positions_w: torch.Tensor,
    state: SonicComplianceCommandState,
    use_target_damper: bool,
) -> None:
    if not isinstance(state, SonicComplianceCommandState):
        raise TypeError("state must be a SonicComplianceCommandState")
    if not isinstance(use_target_damper, bool):
        raise TypeError("use_target_damper must be a bool")
    expected_reference_prefix = (state.num_envs, state.num_future_frames)
    if tuple(reference_positions_w.shape[:2]) != expected_reference_prefix:
        raise ValueError(
            "reference_positions_w env/future dimensions do not match command state"
        )
    if articulation_positions_w.shape[0] != state.num_envs:
        raise ValueError("articulation_positions_w env dimension does not match command state")
    if reference_positions_w.dtype != state.dtype or articulation_positions_w.dtype != state.dtype:
        raise TypeError("position tensors must use command-state dtype")
    if (
        reference_positions_w.device != state.device
        or articulation_positions_w.device != state.device
    ):
        raise ValueError("position tensors must use command-state device")


def build_sonic_compliance_targets(
    *,
    reference_positions_w: torch.Tensor,
    reference_quaternions_wxyz: torch.Tensor,
    articulation_positions_w: torch.Tensor,
    articulation_quaternions_wxyz: torch.Tensor,
    anchor_position_w: torch.Tensor | None,
    anchor_quaternion_wxyz: torch.Tensor | None,
    state: SonicComplianceCommandState,
    use_target_damper: bool = False,
) -> SonicComplianceTargets:
    """Build the unique actor-facing hindsight target from checked inputs.

    The nominal source is the original reference by default.  When the optional
    CHIP target damper is explicitly enabled, its cached command-side target is
    selected only at positively compliant requested sites.  In both cases the
    sole observed target is ``nominal - C * force_on_robot``.  Dense rewards must
    continue to consume the untouched original reference outside this adapter.
    """

    _validate_builder_inputs(
        reference_positions_w=reference_positions_w,
        articulation_positions_w=articulation_positions_w,
        state=state,
        use_target_damper=use_target_damper,
    )
    return _build_sonic_compliance_targets(
        reference_positions_w=reference_positions_w,
        reference_quaternions_wxyz=reference_quaternions_wxyz,
        articulation_positions_w=articulation_positions_w,
        articulation_quaternions_wxyz=articulation_quaternions_wxyz,
        anchor_position_w=anchor_position_w,
        anchor_quaternion_wxyz=anchor_quaternion_wxyz,
        state=state,
        use_target_damper=use_target_damper,
        prevalidated=False,
    )


def build_sonic_compliance_targets_prevalidated(
    *,
    reference_positions_w: torch.Tensor,
    reference_quaternions_wxyz: torch.Tensor,
    articulation_positions_w: torch.Tensor,
    articulation_quaternions_wxyz: torch.Tensor,
    anchor_position_w: torch.Tensor | None,
    anchor_quaternion_wxyz: torch.Tensor | None,
    state: SonicComplianceCommandState,
    use_target_damper: bool = False,
) -> SonicComplianceTargets:
    """Build targets from lifecycle-validated simulator tensors without host sync."""

    return _build_sonic_compliance_targets(
        reference_positions_w=reference_positions_w,
        reference_quaternions_wxyz=reference_quaternions_wxyz,
        articulation_positions_w=articulation_positions_w,
        articulation_quaternions_wxyz=articulation_quaternions_wxyz,
        anchor_position_w=anchor_position_w,
        anchor_quaternion_wxyz=anchor_quaternion_wxyz,
        state=state,
        use_target_damper=use_target_damper,
        prevalidated=True,
    )
