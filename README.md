# inspect-robots-dream

Run [Inspect Robots](https://github.com/robocurve/inspect-robots) evals with
[Dream Labs](https://platform.dreamlabs.ai/docs/)' **Dream-1** robot model as the policy,
called through the Dream API with the `dream` client library.

> [!NOTE]
> Early development. The first pairing is the bimanual YAM (`yam_arms` from
> [inspect-robots-yam](https://github.com/robocurve/inspect-robots-yam)), which Dream-1 runs
> zero-shot. Other arms are not wired up yet; see [`plans/0001`](plans/0001-dream-policy-design.md).

Inspect Robots has two swappable inputs: a `Policy` (the brain) and an `Embodiment` (the
body). This package provides the policy:

- **`dream` policy**: sends each observation (three cameras, both arms' joints, the task
  text) to Dream-1 and returns its 48-row chunk as 14-D absolute joint targets, left arm then
  right, each `[j0 … j5, gripper]` with the gripper 0 closed to 1 open. That is the
  `yam_arms` contract, so `inspect_robots.compat.check_compatibility` passes.

```bash
inspect-robots run --policy dream --embodiment yam_arms -E control_hz=20 \
  --instruction "fold the towel"
```

## Install

```bash
python -m pip install "inspect-robots-dream[ik] @ git+https://github.com/kyeshmz/inspect-robots-dream"
```

- The `ik` extra installs `dreamlabs`, which ships the solver the client needs to add the
  hand poses `dreamlabs/dream-1-beta` reads. It needs Python 3.11–3.13 and brings jax and
  mujoco.
- The service refuses `dreamlabs-client` 0.x (`DA-4000 client_outdated`), so the package
  requires 1.0.3 or newer.

**API key.** Follow step 2 of Dream's
[Getting started](https://platform.dreamlabs.ai/docs/), or the platform's
[Setup API](https://platform.dreamlabs.ai/setup-api) page. Either one writes
`~/.dreamlabs/credentials`, where the client reads your key and the service addresses. Each
successful `get_action` is billed against your credit; failed calls cost nothing.

## How a trial runs

1. `reset()` opens the Dream policy on the first trial. Later trials end the last session
   (`reset_episode()`), so each trial starts a fresh one.
2. `act()` hands the client one tick: each camera stretched to 320×176 and sent as a JPEG at
   quality 85, both arms' state, and the scene's instruction as `prompt`. It then asks for an
   action chunk.
3. Inspect plays every row of the chunk. The arms hold while Dream answers, as in the client's
   `sync` pacing. The last row played goes back as `commanded` on the next call, which anchors
   the client's seam smoothing.

Inspect calls `act()` once per chunk rather than on every camera tick, so the frame window
defaults to a single frame (`obs_window=1`). A window built from frames a whole chunk apart
would misstate how fast things moved.

## Configuration

`DreamPolicy` keyword arguments, all scalars and settable with `-P key=value`:

| Argument | Default | Meaning |
| --- | --- | --- |
| `checkpoint` | `dreamlabs/dream-1-beta` | The model: an id `DreamClient().get_models()` lists, or a fine-tune's |
| `cameras` | `top_cam,left_cam,right_cam` | The embodiment's overhead, left-wrist and right-wrist cameras, in that order |
| `state_key` | `joint_pos` | The packed 14-D state |
| `head` | `joint` | `joint` runs the model's YAM joints; `eef` solves its hand poses to joints |
| `control_hz` | `20` | Row rate the chunk is meant to play at (the YAM station's); also the client's `action_hz` |
| `action_horizon` | `48` | Recorded in the log; the played length is always the reply's |
| `obs_window` | `1` | Frames the client keeps per camera |
| `timeout_ms` | `15000` | Per-request timeout |
| `retry_budget_s` | `60` | How long one `act()` keeps retrying after a retryable refusal |
| `name` | `dream` | The policy's name in eval logs |

**Rates.** `yam_arms` runs at 10 Hz by default. Pass `-E control_hz=20` so that Dream's plan
plays at the speed it was made for. Otherwise the compatibility check warns that the rates
differ.

**A cold model.** The first call to a model that has been idle is refused with
`pool_warming` while it starts, which Dream says can take 25 to 50 minutes. With the default
budget the trial fails after about a minute and reports `pool_warming`. Run it again once
the model is up, or pass `-P retry_budget_s=3600` to wait inside the trial.

## Safety

The policy only proposes targets. Limits, clamps and stops belong to the embodiment and to
Inspect's approvers, as with any other policy. Run `inspect-robots-yam-preflight` and the
YAM rig's own checks before letting Dream move the arms.

## Development

```bash
python -m venv .venv && .venv/bin/python -m pip install -e ".[dev]"
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy                     # --strict over src/inspect_robots_dream
.venv/bin/python -m pytest
```

The suite needs no key, network or robot. The Dream client is injected (`client_factory`),
and an end-to-end test runs a full `eval()` with a fake client on a fake YAM. With
inspect-robots-yam installed, `tests/test_compat.py` also pairs the policy with the real
`yam_arms` declaration.

## License

[MIT](LICENSE)
