"""The bimanual-YAM pairing: Inspect Robots's 14-D ``joint_pos`` against Dream-1's YAM station keys.

Inspect side: the ``yam_arms`` embodiment of inspect-robots-yam declares one flat 14-D vector
for both state and actions, left arm then right arm, each ``[j0 .. j5, gripper]`` in radians
with the gripper normalized 0 (closed) to 1 (open). This module copies that packing rather than
importing it, so the policy installs without the YAM driver stack; ``tests/test_compat.py``
checks the two still agree when inspect-robots-yam is installed.

Dream side: ``dreamlabs/dream-1-beta`` on a YAM reads ``state.<side>_arm`` (6) and
``state.<side>_gripper`` (1, 0 closed to 1 open), three camera windows ``video.overhead``,
``video.wrist_l`` and ``video.wrist_r``, and answers ``q_l_chunk`` / ``q_r_chunk``, each
``[H, 7]``: six joint targets, then the gripper. The gripper conventions already match, so
nothing is rescaled.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)

ARM_DOF = 6
ARM_WIDTH = ARM_DOF + 1  # six joints, then the gripper
TOTAL_DIM = 2 * ARM_WIDTH
LEFT = slice(0, ARM_WIDTH)
RIGHT = slice(ARM_WIDTH, TOTAL_DIM)

DIM_LABELS: tuple[str, ...] = tuple(
    f"{side}_{part}"
    for side in ("left", "right")
    for part in (*(f"j{i}" for i in range(ARM_DOF)), "gripper")
)

# Dream's camera roles, in the order the policy's ``cameras`` argument names Inspect cameras.
ROLES: tuple[str, ...] = ("overhead", "wrist_l", "wrist_r")

# The size Dream-1 reads every frame at (width, height); others are stretched to it.
FRAME_SIZE = (320, 176)

# The YAM body as the Dream client's strict mode checks it (its README's own example).
EMBODIMENT: Mapping[str, Any] = {
    "state": {
        "right_arm": [0, 6],
        "right_gripper": [6, 7],
        "left_arm": [7, 13],
        "left_gripper": [13, 14],
    },
    "action": {
        "right_arm": [0, 6],
        "right_gripper": [6, 7],
        "left_arm": [7, 13],
        "left_gripper": [13, 14],
    },
    "video": {role: f"observation.images.{role}" for role in ROLES},
}

Vec = npt.NDArray[np.float64]


def action_space() -> Box:
    """The 14-D absolute joint action the policy emits; limits are the embodiment's to set."""
    return Box(
        shape=(TOTAL_DIM,),
        semantics=ActionSemantics(
            control_mode="joint_pos",
            rotation_repr="none",
            gripper="continuous",
            frame="base",
            dim_labels=DIM_LABELS,
        ),
    )


def observation_space(cameras: tuple[str, ...], state_key: str) -> ObservationSpace:
    """The cameras and the 14-D state the policy reads, under the embodiment's names."""
    width, height = FRAME_SIZE
    return ObservationSpace(
        cameras=tuple(CameraSpec(name=name, height=height, width=width) for name in cameras),
        state=StateSpec(
            fields=(StateField(key=state_key, shape=(TOTAL_DIM,), unit="rad+normalized"),)
        ),
    )


def dream_state(joint_pos: npt.ArrayLike) -> dict[str, npt.NDArray[np.float32]]:
    """Split the packed 14-D state into Dream's per-arm ``state.*`` keys."""
    vec = np.asarray(joint_pos, dtype=np.float64)
    if vec.shape != (TOTAL_DIM,):
        raise ValueError(f"expected a {TOTAL_DIM}-D joint_pos, got shape {vec.shape}")
    if not np.isfinite(vec).all():
        raise ValueError("joint_pos holds a non-finite value")
    left, right = vec[LEFT].astype(np.float32), vec[RIGHT].astype(np.float32)
    return {
        "state.left_arm": left[:ARM_DOF],
        "state.left_gripper": left[ARM_DOF:],
        "state.right_arm": right[:ARM_DOF],
        "state.right_gripper": right[ARM_DOF:],
    }


def joint_rows(action: Mapping[str, Any]) -> Vec:
    """Pack Dream's ``q_l_chunk`` and ``q_r_chunk`` into ``[H, 14]`` rows, left then right."""
    chunks = []
    for key in ("q_l_chunk", "q_r_chunk"):
        if action.get(key) is None:
            raise ValueError(f"Dream's reply carries no {key!r}")
        chunk = np.asarray(action[key], dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != ARM_WIDTH or chunk.shape[0] == 0:
            raise ValueError(f"Dream's {key!r} has shape {chunk.shape}; expected (H, {ARM_WIDTH})")
        chunks.append(chunk)
    left, right = chunks
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            f"Dream's arm chunks differ in length: {left.shape[0]} and {right.shape[0]}"
        )
    rows = np.concatenate([left, right], axis=1)
    if not np.isfinite(rows).all():
        raise ValueError("Dream's reply holds a non-finite joint target")
    return rows


def commanded(row: npt.ArrayLike) -> dict[str, Vec]:
    """The last packed row the loop commanded, as Dream's per-arm ``commanded`` anchor."""
    vec = np.asarray(row, dtype=np.float64)
    return {"q_l_chunk": vec[LEFT].copy(), "q_r_chunk": vec[RIGHT].copy()}
