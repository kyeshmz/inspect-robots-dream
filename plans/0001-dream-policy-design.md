# 0001: Dream-1 as an Inspect Robots policy

Status: draft (2026-10-21)

## Goal

Register a `dream` policy that runs Dream Labs' Dream-1 through `dream.DreamClient`, so any
embodiment-agnostic Inspect Robots task can be evaluated with Dream-1 on a real arm.

| Robot | Embodiment | Package | Dream arm | Status |
| --- | --- | --- | --- | --- |
| I2RT YAM (bimanual) | `yam_arms` | inspect-robots-yam | `yam` (zero-shot) | Implemented, offline-tested |
| Seeed reBot B601-RS | `b601` | inspect-robots-b601 | arm file from the URDF | Blocked (below) |

## Contract (YAM)

- **Action:** 14-D absolute joint targets, left arm then right, each `[j0 … j5, gripper]`.
  - `ActionSemantics(control_mode="joint_pos", frame="base", gripper="continuous")`.
  - The packing is copied from inspect-robots-yam's `packing.py`. A test compares the two
    whenever inspect-robots-yam is installed.
- **Observation:** `top_cam`, `left_cam` and `right_cam`, mapped to Dream's `video.overhead`,
  `video.wrist_l` and `video.wrist_r`, plus the 14-D `joint_pos`.
  - The state is split into `state.<side>_arm` (6) and `state.<side>_gripper` (1).
  - Both sides use 0 closed to 1 open (inspect-robots-yam plan 0005), so nothing is rescaled.
- **Rate:** `control_hz=20`, the YAM station's rate in Dream's `robots` package.
  - The policy declares it in `PolicyInfo`, so the compatibility check warns against the YAM's
    default of 10 Hz.
- **Dream config:** `strict` on, the YAM `embodiment` block from the client's README, and
  `ik: {arm: yam, head: joint}`.
  - dream-1-beta reads hand poses, and the client adds them.
  - Offline, the real client 1.0.3 accepts every other key. With `ik` set, `policy()` looks the
    model up on the platform, which needs a key.

## Decisions

- **Synchronous `act()`.**
  - Inspect calls `act()` once per chunk and plays the whole chunk. Each call hands the client
    one tick (`set_observation`) and asks at once (`get_action`). No camera thread runs.
  - The arms hold during inference. This is the client's `sync` pacing.
- **`obs_window=1`.**
  - Successive `act()` frames are a chunk apart, about 2.4 s at 48 rows and 20 Hz. Dream
    assumes 0.2 s between kept frames, so a window built from them would misstate motion.
  - A single frame is a still image, which Dream documents as accepted.
  - Restoring the full 9-frame window needs frames from between `act()` calls, so the
    embodiment would have to feed the client (open item).
- **`commanded` anchoring.** The last row of the previous chunk is passed as `commanded`, which
  the client recommends over the measured pose because a real arm trails its command.
- **Retries.**
  - A retryable `DreamServerError` (`rate_limited`, `pool_full`, `pool_warming`, ...),
    `DreamSessionExpired` and `DreamTimeout` are asked again within `retry_budget_s`
    (default 60 s).
  - Everything else is raised, and Inspect reports it as a `PolicyError` with the policy's
    `remedy` line.
- **Construction is inert.** Nothing is imported from `dream`, no credentials are read and
  nothing is sent until the first `reset()`. Importing `dream` starts its log file.

## Not yet verified

- **A live call.** No key is on this machine yet. The offline checks stop at the network.
- **Seam smoothing while the arms hold.** The client blends each chunk using the measured pose,
  the last chunk and the round trip. In sync pacing an arm that has played every row
  "starts from rest". Whether row 0 of a new chunk then continues smoothly from the held pose
  needs a recorded reply to confirm.

## B601-RS: blocked

The B601-RS is an arm Dream-1 runs from its URDF through an arm file (`ik.arm` = path). Four
things are open before it can run:

1. **Tool center point.**
   - Dream's docs require the user to measure and confirm the TCP (`dreamcli tcp`, or by hand)
     before any IK or control loop on an arm from its URDF. A coding agent must not pick or
     estimate it.
   - The arm file's `tcp`, `approach` and `jaw` stay empty until the user has confirmed them.
2. **DM versus RS.**
   - Dream ships a reBot model, `seeed_faas` / `b601`, but it is the B601-**DM**. Its URDF
     takes the DM motor angles that robot reports, with its own signs and soft limits.
   - inspect-robots-b601 speaks the RS's raw RobStride motor radians. Unless the two are shown
     to match joint by joint, the RS needs its own arm file.
3. **One arm.**
   - Dream's arm files and IK run both sides (`SIDES = ("right", "left")`), and the base models
     are bimanual.
   - The documentation does not say how a single-arm robot is sent: an idle second arm, or a
     mirrored copy.
4. **Rate and speed.**
   - The `b601` embodiment walks at 0.05 rad/s and steps at 0.2–2 Hz, because it was designed
     for the LLM agent.
   - Dream plans 48 rows at about 20–30 Hz. Played one row per step, a chunk would run more
     than 10× slower than planned.
   - Pairing the two needs either a faster mode on the embodiment or a policy-side resample.
     That is a safety decision for the B601 owner.
