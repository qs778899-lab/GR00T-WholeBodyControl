"""Explicit transforms between world and an adapter-selected common frame."""

from __future__ import annotations

import torch


def _align_trailing_components(tensor: torch.Tensor, target_ndim: int) -> torch.Tensor:
    if tensor.ndim > target_ndim:
        raise ValueError("frame tensor has more axes than the target tensor")
    if tensor.ndim == target_ndim:
        return tensor
    return tensor.reshape(
        *tensor.shape[:-1],
        *([1] * (target_ndim - tensor.ndim)),
        tensor.shape[-1],
    )


def _validate_quaternion(quaternion_wxyz: torch.Tensor) -> None:
    if not isinstance(quaternion_wxyz, torch.Tensor):
        raise TypeError("quaternion_wxyz must be a tensor")
    if quaternion_wxyz.shape[-1] != 4:
        raise ValueError("quaternion_wxyz must end in four WXYZ components")
    if not quaternion_wxyz.is_floating_point() or quaternion_wxyz.is_complex():
        raise TypeError("quaternion_wxyz must have a real floating dtype")
    if not torch.isfinite(quaternion_wxyz).all():
        raise ValueError("quaternion_wxyz must be finite")
    norm = torch.linalg.vector_norm(quaternion_wxyz, dim=-1)
    if not torch.allclose(norm, torch.ones_like(norm), rtol=1.0e-5, atol=1.0e-6):
        raise ValueError("quaternion_wxyz must be normalized")


def rotate_vectors_wxyz(
    quaternion_wxyz: torch.Tensor,
    vectors: torch.Tensor,
) -> torch.Tensor:
    """Rotate vectors using normalized WXYZ quaternions with broadcasting."""

    _validate_quaternion(quaternion_wxyz)
    if not isinstance(vectors, torch.Tensor):
        raise TypeError("vectors must be a tensor")
    if vectors.shape[-1] != 3:
        raise ValueError("vectors must end in three Cartesian components")
    if vectors.dtype != quaternion_wxyz.dtype or vectors.device != quaternion_wxyz.device:
        raise ValueError("vectors and quaternion_wxyz must share dtype and device")
    if not torch.isfinite(vectors).all():
        raise ValueError("vectors must be finite")

    return _rotate_vectors_wxyz_unchecked(quaternion_wxyz, vectors)


def _rotate_vectors_wxyz_unchecked(
    quaternion_wxyz: torch.Tensor,
    vectors: torch.Tensor,
) -> torch.Tensor:
    """Rotate vectors without CUDA value reductions; caller owns validation."""

    quaternion_wxyz = _align_trailing_components(quaternion_wxyz, vectors.ndim)
    vector_part, vectors = torch.broadcast_tensors(quaternion_wxyz[..., 1:], vectors)
    scalar_part = torch.broadcast_to(quaternion_wxyz[..., :1], (*vectors.shape[:-1], 1))
    first_cross = torch.cross(vector_part, vectors, dim=-1)
    return vectors + 2.0 * (
        scalar_part * first_cross + torch.cross(vector_part, first_cross, dim=-1)
    )


def common_to_world_vectors(
    vectors_common: torch.Tensor,
    common_frame_quaternion_world_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Rotate common-frame vectors into world; translation never affects force."""

    return rotate_vectors_wxyz(common_frame_quaternion_world_wxyz, vectors_common)


def _common_to_world_vectors_unchecked(
    vectors_common: torch.Tensor,
    common_frame_quaternion_world_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Fast common-to-world rotation for simulator-owned finite tensors."""

    return _rotate_vectors_wxyz_unchecked(
        common_frame_quaternion_world_wxyz,
        vectors_common,
    )


def world_to_common_positions(
    positions_world: torch.Tensor,
    common_frame_origin_world: torch.Tensor,
    common_frame_quaternion_world_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Express world positions in one explicit common Cartesian frame."""

    if positions_world.dtype != common_frame_origin_world.dtype:
        raise ValueError("positions and common-frame origin must share dtype")
    if positions_world.device != common_frame_origin_world.device:
        raise ValueError("positions and common-frame origin must share device")
    if common_frame_origin_world.shape[-1] != 3:
        raise ValueError("common-frame origin must end in three Cartesian components")
    origin = _align_trailing_components(common_frame_origin_world, positions_world.ndim)
    translated = positions_world - origin
    inverse = _align_trailing_components(
        common_frame_quaternion_world_wxyz,
        positions_world.ndim,
    ).clone()
    inverse[..., 1:] = -inverse[..., 1:]
    return rotate_vectors_wxyz(inverse, translated)


def _world_to_common_positions_unchecked(
    positions_world: torch.Tensor,
    common_frame_origin_world: torch.Tensor,
    common_frame_quaternion_world_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Fast world-to-common transform for simulator-owned finite tensors."""

    origin = _align_trailing_components(common_frame_origin_world, positions_world.ndim)
    translated = positions_world - origin
    inverse = _align_trailing_components(
        common_frame_quaternion_world_wxyz,
        positions_world.ndim,
    ).clone()
    inverse[..., 1:] = -inverse[..., 1:]
    return _rotate_vectors_wxyz_unchecked(inverse, translated)


def _world_to_body_vectors_unchecked(
    vectors_world: torch.Tensor,
    body_quaternion_world_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Express world vectors in each body's current link frame without reductions."""

    inverse = _align_trailing_components(
        body_quaternion_world_wxyz,
        vectors_world.ndim,
    ).clone()
    inverse[..., 1:] = -inverse[..., 1:]
    return _rotate_vectors_wxyz_unchecked(inverse, vectors_world)
