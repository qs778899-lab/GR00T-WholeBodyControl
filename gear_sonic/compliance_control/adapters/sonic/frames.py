"""Pure-Torch transforms at the SONIC compliance boundary.

Isaac Lab publishes positions and forces in the simulation world frame.  This
module converts both quantities into the same structured frame declared by the
portable compliance contract.  Quaternions use Isaac Lab's ``wxyz`` order.
"""

from __future__ import annotations

import torch

from ...core import CartesianFrameKind, CartesianFrameSpec, CartesianRotation


def _validate_cartesian_tensor(tensor: torch.Tensor, *, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if tensor.ndim < 2 or tensor.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [..., xyz]")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")


def _validate_anchor(
    anchor: torch.Tensor,
    reference: torch.Tensor,
    *,
    name: str,
    final_dimension: int,
) -> None:
    if not isinstance(anchor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not anchor.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if anchor.shape[-1] != final_dimension:
        raise ValueError(f"{name} final dimension must be {final_dimension}")
    if anchor.dtype != reference.dtype:
        raise TypeError(f"{name} dtype {anchor.dtype} does not match {reference.dtype}")
    if anchor.device != reference.device:
        raise ValueError(f"{name} device {anchor.device} does not match {reference.device}")
    if not torch.isfinite(anchor).all():
        raise ValueError(f"{name} must contain only finite values")


def _expand_leading_dimensions(anchor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    expanded = anchor
    while expanded.ndim < target.ndim:
        expanded = expanded.unsqueeze(-2)
    try:
        return expanded.expand(*target.shape[:-1], expanded.shape[-1])
    except RuntimeError as exc:
        raise ValueError(
            f"anchor shape {tuple(anchor.shape)} cannot broadcast to {tuple(target.shape)}"
        ) from exc


def _normalize_quaternion_prevalidated(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.vector_norm(quaternion_wxyz, dim=-1, keepdim=True)
    return quaternion_wxyz / norms.clamp_min(torch.finfo(quaternion_wxyz.dtype).eps)


def _normalize_quaternion(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.vector_norm(quaternion_wxyz, dim=-1, keepdim=True)
    if (norms <= torch.finfo(quaternion_wxyz.dtype).eps).any():
        raise ValueError("anchor_quaternion_wxyz must contain non-zero quaternions")
    return quaternion_wxyz / norms


def _yaw_quaternion(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Return the Z-up yaw component of a normalized ``wxyz`` quaternion."""

    w, x, y, z = quaternion_wxyz.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    half_yaw = 0.5 * yaw
    zeros = torch.zeros_like(half_yaw)
    return torch.stack((torch.cos(half_yaw), zeros, zeros, torch.sin(half_yaw)), dim=-1)


def _quat_apply(quaternion_wxyz: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    q_vector = quaternion_wxyz[..., 1:]
    q_scalar = quaternion_wxyz[..., :1]
    uv = torch.linalg.cross(q_vector, vectors, dim=-1)
    uuv = torch.linalg.cross(q_vector, uv, dim=-1)
    return vectors + 2.0 * (q_scalar * uv + uuv)


def _quat_apply_inverse(quaternion_wxyz: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    conjugate = torch.cat((quaternion_wxyz[..., :1], -quaternion_wxyz[..., 1:]), dim=-1)
    return _quat_apply(conjugate, vectors)


def quaternion_rotate_wxyz(
    quaternion_wxyz: torch.Tensor,
    vectors: torch.Tensor,
) -> torch.Tensor:
    """Rotate local vectors by matching/broadcastable normalized ``wxyz`` quaternions."""

    _validate_cartesian_tensor(vectors, name="vectors")
    _validate_anchor(
        quaternion_wxyz,
        vectors,
        name="quaternion_wxyz",
        final_dimension=4,
    )
    quaternion = _expand_leading_dimensions(_normalize_quaternion(quaternion_wxyz), vectors)
    return _quat_apply(quaternion, vectors)


def quaternion_rotate_wxyz_prevalidated(
    quaternion_wxyz: torch.Tensor,
    vectors: torch.Tensor,
) -> torch.Tensor:
    """Rotate lifecycle-validated tensors without CUDA scalar extraction."""

    quaternion = _expand_leading_dimensions(
        _normalize_quaternion_prevalidated(quaternion_wxyz),
        vectors,
    )
    return _quat_apply(quaternion, vectors)


def quaternion_rotate_inverse_wxyz(
    quaternion_wxyz: torch.Tensor,
    vectors: torch.Tensor,
) -> torch.Tensor:
    """Rotate world vectors into matching normalized ``wxyz`` body frames."""

    _validate_cartesian_tensor(vectors, name="vectors")
    _validate_anchor(
        quaternion_wxyz,
        vectors,
        name="quaternion_wxyz",
        final_dimension=4,
    )
    quaternion = _expand_leading_dimensions(_normalize_quaternion(quaternion_wxyz), vectors)
    return _quat_apply_inverse(quaternion, vectors)


def quaternion_rotate_inverse_wxyz_prevalidated(
    quaternion_wxyz: torch.Tensor,
    vectors: torch.Tensor,
) -> torch.Tensor:
    """Inverse-rotate lifecycle-validated tensors without host synchronization."""

    quaternion = _expand_leading_dimensions(
        _normalize_quaternion_prevalidated(quaternion_wxyz),
        vectors,
    )
    return _quat_apply_inverse(quaternion, vectors)


def _frame_rotation(
    frame: CartesianFrameSpec,
    anchor_quaternion_wxyz: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    if frame.kind is CartesianFrameKind.WORLD:
        return None
    if anchor_quaternion_wxyz is None:
        raise ValueError(f"{frame.kind.value} requires anchor_quaternion_wxyz")
    _validate_anchor(
        anchor_quaternion_wxyz,
        reference,
        name="anchor_quaternion_wxyz",
        final_dimension=4,
    )
    quaternion = _normalize_quaternion(anchor_quaternion_wxyz)
    if frame.rotation is CartesianRotation.YAW_ONLY:
        quaternion = _yaw_quaternion(quaternion)
    elif frame.rotation is not CartesianRotation.FULL_3D:
        raise ValueError(f"unsupported local rotation mode: {frame.rotation.value}")
    return _expand_leading_dimensions(quaternion, reference)


def _frame_rotation_prevalidated(
    frame: CartesianFrameSpec,
    anchor_quaternion_wxyz: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    if frame.kind is CartesianFrameKind.WORLD:
        return None
    if anchor_quaternion_wxyz is None:
        raise ValueError(f"{frame.kind.value} requires anchor_quaternion_wxyz")
    quaternion = _normalize_quaternion_prevalidated(anchor_quaternion_wxyz)
    if frame.rotation is CartesianRotation.YAW_ONLY:
        quaternion = _yaw_quaternion(quaternion)
    elif frame.rotation is not CartesianRotation.FULL_3D:
        raise ValueError(f"unsupported local rotation mode: {frame.rotation.value}")
    return _expand_leading_dimensions(quaternion, reference)


def world_positions_to_frame(
    positions_w: torch.Tensor,
    *,
    frame: CartesianFrameSpec,
    anchor_position_w: torch.Tensor | None = None,
    anchor_quaternion_wxyz: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert world positions to ``frame`` without mutating the source tensor."""

    _validate_cartesian_tensor(positions_w, name="positions_w")
    if not isinstance(frame, CartesianFrameSpec):
        raise TypeError("frame must be a CartesianFrameSpec")
    if frame.kind is CartesianFrameKind.WORLD:
        return positions_w.clone()
    if anchor_position_w is None:
        raise ValueError(f"{frame.kind.value} requires anchor_position_w")
    _validate_anchor(
        anchor_position_w,
        positions_w,
        name="anchor_position_w",
        final_dimension=3,
    )
    anchor_position = _expand_leading_dimensions(anchor_position_w, positions_w)
    rotation = _frame_rotation(frame, anchor_quaternion_wxyz, positions_w)
    assert rotation is not None
    return _quat_apply_inverse(rotation, positions_w - anchor_position)


def world_positions_to_frame_prevalidated(
    positions_w: torch.Tensor,
    *,
    frame: CartesianFrameSpec,
    anchor_position_w: torch.Tensor | None = None,
    anchor_quaternion_wxyz: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert lifecycle-validated world positions without host synchronization."""

    if frame.kind is CartesianFrameKind.WORLD:
        return positions_w.clone()
    if anchor_position_w is None:
        raise ValueError(f"{frame.kind.value} requires anchor_position_w")
    anchor_position = _expand_leading_dimensions(anchor_position_w, positions_w)
    rotation = _frame_rotation_prevalidated(frame, anchor_quaternion_wxyz, positions_w)
    assert rotation is not None
    return _quat_apply_inverse(rotation, positions_w - anchor_position)


def frame_positions_to_world(
    positions_frame: torch.Tensor,
    *,
    frame: CartesianFrameSpec,
    anchor_position_w: torch.Tensor | None = None,
    anchor_quaternion_wxyz: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert structured-frame positions back to the world frame."""

    _validate_cartesian_tensor(positions_frame, name="positions_frame")
    if not isinstance(frame, CartesianFrameSpec):
        raise TypeError("frame must be a CartesianFrameSpec")
    if frame.kind is CartesianFrameKind.WORLD:
        return positions_frame.clone()
    if anchor_position_w is None:
        raise ValueError(f"{frame.kind.value} requires anchor_position_w")
    _validate_anchor(
        anchor_position_w,
        positions_frame,
        name="anchor_position_w",
        final_dimension=3,
    )
    anchor_position = _expand_leading_dimensions(anchor_position_w, positions_frame)
    rotation = _frame_rotation(frame, anchor_quaternion_wxyz, positions_frame)
    assert rotation is not None
    return _quat_apply(rotation, positions_frame) + anchor_position


def world_vectors_to_frame(
    vectors_w: torch.Tensor,
    *,
    frame: CartesianFrameSpec,
    anchor_quaternion_wxyz: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rotate world vectors into ``frame``; vectors are never translated."""

    _validate_cartesian_tensor(vectors_w, name="vectors_w")
    if not isinstance(frame, CartesianFrameSpec):
        raise TypeError("frame must be a CartesianFrameSpec")
    rotation = _frame_rotation(frame, anchor_quaternion_wxyz, vectors_w)
    if rotation is None:
        return vectors_w.clone()
    return _quat_apply_inverse(rotation, vectors_w)


def world_vectors_to_frame_prevalidated(
    vectors_w: torch.Tensor,
    *,
    frame: CartesianFrameSpec,
    anchor_quaternion_wxyz: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rotate lifecycle-validated world vectors without host synchronization."""

    rotation = _frame_rotation_prevalidated(frame, anchor_quaternion_wxyz, vectors_w)
    if rotation is None:
        return vectors_w.clone()
    return _quat_apply_inverse(rotation, vectors_w)


def frame_vectors_to_world(
    vectors_frame: torch.Tensor,
    *,
    frame: CartesianFrameSpec,
    anchor_quaternion_wxyz: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rotate structured-frame vectors back into the world frame."""

    _validate_cartesian_tensor(vectors_frame, name="vectors_frame")
    if not isinstance(frame, CartesianFrameSpec):
        raise TypeError("frame must be a CartesianFrameSpec")
    rotation = _frame_rotation(frame, anchor_quaternion_wxyz, vectors_frame)
    if rotation is None:
        return vectors_frame.clone()
    return _quat_apply(rotation, vectors_frame)
