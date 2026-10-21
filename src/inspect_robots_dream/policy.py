"""Dream-1 as an Inspect Robots policy, called through the Dream API (``dream.DreamClient``).

Inspect's rollout is synchronous: it calls ``act()`` with one observation, plays the whole
chunk, then calls again. The Dream client is built for a camera loop that hands it every tick
while ``get_action()`` runs on a thread of its own; here each ``act()`` hands it one tick and
asks at once. Two consequences, both deliberate:

- The frame window defaults to one frame (``obs_window=1``). Frames from successive ``act()``
  calls are a whole chunk apart, and a window of them would tell the model that seconds of
  motion happened in a fraction of one. A single frame is a still image, which Dream accepts.
- The arms hold while Dream answers, which is the client's own ``sync`` pacing: every row of
  each chunk is played, and the last row goes back to the client as ``commanded``.

Construction is inert: no credentials are read and nothing is sent until the first
``reset()``, so the registry, ``inspect-robots doctor`` and the compatibility check construct
it freely. The client is injected (``client_factory``), so the suite runs with no key and no
network.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Protocol

from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.types import Action, ActionChunk, ImageArray, Observation

from inspect_robots_dream import yam
from inspect_robots_dream.frames import encode_jpeg

DEFAULT_CHECKPOINT = "dreamlabs/dream-1-beta"
HEADS = ("joint", "eef")


class DreamClientLike(Protocol):
    """The part of ``dream.DreamClient`` the policy calls."""

    @property
    def last_timing(self) -> Mapping[str, Any] | None:
        """Where the last ``get_action()``'s milliseconds went."""
        ...

    @property
    def session_id(self) -> str | None:
        """The session the model answers this episode under."""
        ...

    def policy(self, config: dict[str, Any]) -> None:
        """Choose the checkpoint; the first ``get_action()`` opens the session."""
        ...

    def set_observation(
        self,
        frames: Mapping[str, str],
        observation: Mapping[str, Any],
        *,
        timestamp: float | None = None,
    ) -> None:
        """Hand the client one camera tick."""
        ...

    def get_action(self, *, commanded: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Send the kept frames with the newest state and return the action chunk."""
        ...

    def reset_episode(self) -> None:
        """End the session and empty the frame window."""
        ...

    def close(self) -> None:
        """Release the session and the connection."""
        ...


def _default_client() -> DreamClientLike:  # pragma: no cover - needs a key and the network
    import dream

    return dream.DreamClient()


class DreamPolicy:
    """Dream-1 driving the bimanual YAM (``yam_arms`` from inspect-robots-yam)."""

    RUNTIME_REQUIREMENTS: ClassVar[Mapping[str, str]] = {
        "dream": "python -m pip install inspect-robots-dream",
        "dream_ik": "python -m pip install 'inspect-robots-dream[ik]'",
    }

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        cameras: str = "top_cam,left_cam,right_cam",
        state_key: str = "joint_pos",
        head: str = "joint",
        control_hz: float = 20.0,
        action_horizon: int = 48,
        obs_window: int = 1,
        timeout_ms: int = 15000,
        retry_budget_s: float = 60.0,
        name: str = "dream",
        *,
        client_factory: Callable[[], DreamClientLike] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Declare the policy; nothing is read or sent until the first ``reset()``.

        ``cameras`` names the embodiment's overhead, left-wrist and right-wrist cameras, in
        that order. ``control_hz`` is the row rate each chunk is meant to be played at (the
        YAM station's 20 Hz); it is also the client's ``action_hz``. ``retry_budget_s`` bounds
        how long one ``act()`` keeps asking again after a retryable refusal; a model that is
        still starting (``pool_warming``) can take far longer, so raise it for a cold model.
        """
        names = tuple(part.strip() for part in cameras.split(","))
        if len(names) != len(yam.ROLES) or not all(names) or len(set(names)) != len(names):
            raise ValueError(
                f"cameras must name {len(yam.ROLES)} distinct cameras for "
                f"{', '.join(yam.ROLES)}, got {cameras!r}"
            )
        if head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}, got {head!r}")
        if not control_hz > 0:
            raise ValueError(f"control_hz must be > 0, got {control_hz!r}")
        if obs_window < 1:
            raise ValueError(f"obs_window must be >= 1, got {obs_window!r}")
        if timeout_ms < 1:
            raise ValueError(f"timeout_ms must be >= 1, got {timeout_ms!r}")
        if not retry_budget_s >= 0:
            raise ValueError(f"retry_budget_s must be >= 0, got {retry_budget_s!r}")

        self._cameras = names
        self._state_key = state_key
        self._control_hz = float(control_hz)
        self._retry_budget_s = float(retry_budget_s)
        self._client_factory = client_factory if client_factory is not None else _default_client
        self._clock = clock
        self._sleep = sleep
        self._client: DreamClientLike | None = None
        self._instruction = ""
        self._commanded: dict[str, Any] | None = None
        self.num_inferences = 0
        self.dream_config: dict[str, Any] = {
            "checkpoint": checkpoint,
            "timeout_ms": int(timeout_ms),
            "strict": True,
            "action_hz": self._control_hz,
            "obs_window": int(obs_window),
            # dream-1-beta reads each hand's pose; the client adds them from YAM's kinematics.
            "ik": {"arm": "yam", "head": head},
            "embodiment": yam.EMBODIMENT,
        }
        self.info = PolicyInfo(
            name=name,
            action_space=yam.action_space(),
            observation_space=yam.observation_space(names, state_key),
            control_hz=self._control_hz,
            checkpoint=checkpoint,
        )
        self.config = PolicyConfig(action_horizon=action_horizon)

    @property
    def remedy(self) -> str:
        """Recovery line Inspect prints after a connection failure (read with ``getattr``)."""
        return (
            "check the Dream API key in ~/.dreamlabs/credentials "
            "(https://platform.dreamlabs.ai/setup-api) and that this machine reaches "
            "api.dreamlabs.ai"
        )

    def reset(self, scene: Scene) -> None:
        """Start an episode: open the Dream policy on first use, else end the last session."""
        self._instruction = scene.instruction or ""
        self._commanded = None
        self.num_inferences = 0
        if self._client is not None:
            self._client.reset_episode()
            return
        client = self._client_factory()
        try:
            # A bad config or a missing IK solver is raised here, before the arms move.
            client.policy(self.dream_config)
        except BaseException:
            client.close()
            raise
        self._client = client

    def act(self, observation: Observation) -> ActionChunk:
        """Send this observation to Dream-1 and return its chunk as 14-D joint targets."""
        client = self._client
        if client is None:
            raise RuntimeError("act() before reset(): the Dream session opens in reset()")
        frames = {
            f"video.{role}": encode_jpeg(self._frame(observation, camera), yam.FRAME_SIZE)
            for role, camera in zip(yam.ROLES, self._cameras, strict=True)
        }
        if self._state_key not in observation.state:
            raise ValueError(f"observation has no state {self._state_key!r}")
        sent = {
            **yam.dream_state(observation.state[self._state_key]),
            "prompt": observation.instruction or self._instruction,
        }
        client.set_observation(frames, sent)
        started = time.perf_counter()
        reply = self._ask(client)
        latency = time.perf_counter() - started
        rows = yam.joint_rows(reply)
        self._commanded = yam.commanded(rows[-1])
        self.num_inferences += 1
        return ActionChunk(
            actions=[Action(data=row.copy()) for row in rows],
            control_hz=self._control_hz,
            inference_latency_s=latency,
            meta={"dream_session": client.session_id, "dream_timing": client.last_timing},
        )

    def close(self) -> None:
        """End the Dream session and drop the connection; safe to call twice."""
        client, self._client = self._client, None
        if client is not None:
            client.close()

    def __enter__(self) -> DreamPolicy:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _frame(self, observation: Observation, camera: str) -> ImageArray:
        frame = observation.images.get(camera)
        if frame is None:
            raise ValueError(f"observation has no frame from camera {camera!r}")
        return frame

    def _ask(self, client: DreamClientLike) -> dict[str, Any]:
        """``get_action()``, asked again after retryable refusals until the budget runs out."""
        import dream  # Lazily: importing it starts the client's log file.

        deadline = self._clock() + self._retry_budget_s
        while True:
            try:
                return client.get_action(commanded=self._commanded)
            except (dream.DreamSessionExpired, dream.DreamTimeout) as exc:
                # Expired: the new session is already adopted. Timeout: the connection was dropped.
                error: Exception = exc
                wait = 0.0
            except dream.DreamServerError as exc:
                if not exc.retryable:
                    raise
                error = exc
                wait = exc.retry_after_s if exc.retry_after_s is not None else 1.0
            if self._clock() + wait > deadline:
                raise error
            self._sleep(wait)
