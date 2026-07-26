"""IsaacLab-free tensor contracts shared by SONIC manager adapters and tests."""

from __future__ import annotations

import torch

from ...core import select_reference


def condition_from_command(command) -> torch.Tensor:
    """Return the public three-value condition without reading privileged state."""

    return command.command


def threshold_from_command(command) -> torch.Tensor:
    """Return the critic-only scalar threshold."""

    return command.state.force_threshold_n.unsqueeze(-1)


def current_site_force_from_command(command) -> torch.Tensor:
    """Flatten the actually applied current-frame site force in common coordinates."""

    return command.state.force_common_future[:, 0].reshape(command.num_envs, -1)


def site_mask_from_command(command) -> torch.Tensor:
    """Return the critic-only site mask with the command state's floating dtype."""

    return command.state.active_site_mask.to(command.state.dtype)


def select_yielded_site_reference(
    original_reference: torch.Tensor,
    compliant_reference: torch.Tensor,
    active_site_mask: torch.Tensor,
    enabled: torch.Tensor,
) -> torch.Tensor:
    """Checked public selector used by offline analysis and contract tests."""

    return select_reference(
        original_reference,
        compliant_reference,
        active_site_mask,
        enabled=enabled,
    )


def _select_yielded_site_reference_unchecked(
    original_reference: torch.Tensor,
    compliant_reference: torch.Tensor,
    active_site_mask: torch.Tensor,
    enabled: torch.Tensor,
) -> torch.Tensor:
    """No-sync selector for validated command-owned current-frame buffers."""

    selected_mask = active_site_mask & enabled.unsqueeze(-1)
    return torch.where(
        selected_mask.unsqueeze(-1),
        compliant_reference,
        original_reference,
    )


def endpoint_position_errors(
    original_reference: torch.Tensor,
    compliant_reference: torch.Tensor,
    current_reference: torch.Tensor,
    active_site_mask: torch.Tensor,
    enabled: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return selected-target and original-target errors per configured site."""

    selected_reference = _select_yielded_site_reference_unchecked(
        original_reference,
        compliant_reference,
        active_site_mask,
        enabled,
    )
    selected_error = torch.linalg.vector_norm(
        current_reference - selected_reference,
        dim=-1,
    )
    original_error = torch.linalg.vector_norm(
        current_reference - original_reference,
        dim=-1,
    )
    return selected_error, original_error


def endpoint_position_errors_from_state(
    state,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read current-aligned tensors from a command state contract.

    Only future index zero is aligned with the current measured endpoint.  The
    returned tensors retain site order and have shape ``[num_envs, num_sites]``.
    """

    return endpoint_position_errors(
        state.original_reference_common[:, 0],
        state.compliant_reference_common[:, 0],
        state.current_reference_common,
        state.active_site_mask,
        state.enabled,
    )


def current_endpoint_position_errors_from_command(
    command,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read current physics state locally instead of a prior command-update cache."""

    site_state = command._site_tracking_state()
    return endpoint_position_errors(
        site_state.original_reference_common[:, 0],
        site_state.compliant_reference_common[:, 0],
        site_state.current_reference_common,
        command.state.active_site_mask,
        command.state.enabled,
    )


def quaternion_error_magnitude_wxyz(
    reference_quaternion: torch.Tensor,
    current_quaternion: torch.Tensor,
) -> torch.Tensor:
    """Shortest geodesic angle; invariant to a shared anchor-frame rotation."""

    absolute_dot = torch.sum(reference_quaternion * current_quaternion, dim=-1).abs()
    return 2.0 * torch.acos(absolute_dot.clamp(max=1.0))


def gated_mean_gaussian_reward(
    error_per_site: torch.Tensor,
    enabled: torch.Tensor,
    std: float,
) -> torch.Tensor:
    """Reduce per-site errors while making every disabled environment exact zero."""

    enabled = enabled.to(torch.bool)
    masked_error = torch.where(
        enabled.unsqueeze(-1),
        error_per_site,
        torch.zeros_like(error_per_site),
    )
    reward = torch.exp(-masked_error.square().mean(dim=-1) / (std * std))
    return torch.where(enabled, reward, torch.zeros_like(reward))
