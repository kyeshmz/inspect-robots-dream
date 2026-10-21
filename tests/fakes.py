"""Stand-ins for the Dream client and the YAM rig; nothing here needs a key, a network or arms."""

import numpy as np
from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.types import Observation, StepResult

from inspect_robots_dream import yam

CAMERAS = ("top_cam", "left_cam", "right_cam")


def reply(rows=48, left=0.1, right=0.2):
    """A Dream YAM reply: ``rows`` rows per arm, six joints then the gripper."""
    return {
        "q_l_chunk": np.full((rows, 7), left).tolist(),
        "q_r_chunk": np.full((rows, 7), right).tolist(),
        "q_l": [0.0] * 7,
        "q_r": [0.0] * 7,
    }


class FakeClient:
    """Records every call; ``get_action`` answers from ``replies`` in order, raising exceptions."""

    def __init__(self, replies=()):
        self.replies = list(replies)
        self.config = None
        self.ticks = []
        self.asked = []
        self.resets = 0
        self.closed = False
        self.last_timing = {"rtt_ms": 320.0}
        self.session_id = "ses_fake"

    def policy(self, config):
        self.config = config

    def set_observation(self, frames, observation, *, timestamp=None):
        self.ticks.append((dict(frames), dict(observation)))

    def get_action(self, *, commanded=None):
        self.asked.append(commanded)
        answer = self.replies.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def reset_episode(self):
        self.resets += 1

    def close(self):
        self.closed = True


class FakeClock:
    """A monotonic clock that only moves when the policy sleeps."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def yam_info(height=224, width=224):
    """The ``yam_arms`` declaration as inspect-robots-yam makes it (joints, absolute)."""
    return EmbodimentInfo(
        name="yam_arms",
        action_space=Box(
            shape=(14,),
            low=np.full(14, -3.0),
            high=np.full(14, 3.0),
            semantics=ActionSemantics(
                control_mode="joint_pos",
                rotation_repr="none",
                gripper="continuous",
                frame="base",
                dim_labels=yam.DIM_LABELS,
            ),
        ),
        observation_space=ObservationSpace(
            cameras=tuple(CameraSpec(name=n, height=height, width=width) for n in CAMERAS),
            state=StateSpec(
                fields=(StateField(key="joint_pos", shape=(14,), unit="rad+normalized"),)
            ),
        ),
        control_hz=20.0,
        is_simulated=True,
    )


def observation(joint_pos=None, instruction=None, height=224, width=224):
    """One YAM observation: three RGB frames and the packed 14-D state."""
    state = np.arange(14, dtype=np.float64) / 10 if joint_pos is None else joint_pos
    images = {
        name: np.full((height, width, 3), 40 * (i + 1), dtype=np.uint8)
        for i, name in enumerate(CAMERAS)
    }
    return Observation(images=images, state={"joint_pos": state}, instruction=instruction)


class FakeYam:
    """A YAM that reaches every target at once and reports it back as its state."""

    def __init__(self):
        self.info = yam_info()
        self.targets = []
        self.closed = False

    def reset(self, scene, *, seed=None):
        return observation(np.zeros(14), scene.instruction)

    def step(self, action):
        self.targets.append(np.asarray(action.data).copy())
        return StepResult(observation=observation(np.asarray(action.data).copy()))

    def close(self):
        self.closed = True
