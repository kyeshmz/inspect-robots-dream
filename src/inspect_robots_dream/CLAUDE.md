# `inspect_robots_dream` package — module map

One Inspect Robots policy over Dream Labs' `dream.DreamClient`. The package is
`mypy --strict` clean and ships `py.typed`.

## Modules

| Module | Responsibility |
|--------|----------------|
| `policy.py` | `DreamPolicy` (`dream`). Construction declares `.info` only. The first `reset()` makes the client and calls `policy(config)`; later resets call `reset_episode()`. `act()` encodes the three cameras, splits the state, calls `set_observation` then `get_action` (retrying retryable refusals within `retry_budget_s`), and returns the reply as a 14-D `ActionChunk`. `DreamClientLike` is the slice of the client it calls. |
| `yam.py` | The bimanual-YAM pairing: the 14-D packing (copied from inspect-robots-yam), the Inspect spaces, the Dream `embodiment` block, `dream_state()` and `joint_rows()`. |
| `frames.py` | `encode_jpeg()`: stretch an RGB frame to Dream's 320×176 and base64 its JPEG at quality 85. |

## Key invariants

- **Construction is inert:** `DreamPolicy()` imports nothing from `dream`, reads no
  credentials and sends nothing, so the registry, `inspect-robots doctor` and
  compatibility checks construct it freely.
- **The client is injected:** `client_factory`, `clock` and `sleep` are seams, so the suite
  runs with no key, network or robot.
- **The TCP of a URDF arm is never picked by an agent:** an arm file's `tcp`, `approach` and
  `jaw` come from the user (`dreamcli tcp`). See `plans/0001`.
