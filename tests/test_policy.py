import base64
import io
import unittest

import dream
import numpy as np
from inspect_robots.scene import Scene
from PIL import Image

from fakes import FakeClient, FakeClock, observation, reply
from inspect_robots_dream import DreamPolicy, yam

SCENE = Scene(id="towel", instruction="fold the towel")


def opened(client, **kwargs):
    policy = DreamPolicy(client_factory=lambda: client, **kwargs)
    policy.reset(SCENE)
    return policy


class ConstructionTests(unittest.TestCase):
    def test_construction_touches_no_client(self):
        made = []
        DreamPolicy(client_factory=lambda: made.append(1) or FakeClient())
        self.assertEqual(made, [])

    def test_info_declares_the_yam_contract(self):
        policy = DreamPolicy()
        self.assertEqual(policy.info.name, "dream")
        self.assertEqual(policy.info.checkpoint, "dreamlabs/dream-1-beta")
        self.assertEqual(policy.info.action_space.shape, (14,))
        self.assertEqual(policy.info.action_space.semantics.control_mode, "joint_pos")
        self.assertEqual(policy.info.action_space.semantics.dim_labels, yam.DIM_LABELS)
        self.assertEqual(
            policy.info.observation_space.camera_names, {"top_cam", "left_cam", "right_cam"}
        )
        self.assertEqual(policy.info.observation_space.state_keys, {"joint_pos"})
        self.assertEqual(policy.info.control_hz, 20.0)
        self.assertEqual(policy.config.action_horizon, 48)

    def test_bad_arguments_are_refused(self):
        for kwargs in (
            {"cameras": "top_cam,left_cam"},
            {"cameras": "top_cam,top_cam,right_cam"},
            {"cameras": "top_cam,,right_cam"},
            {"head": "wrist"},
            {"control_hz": 0},
            {"obs_window": 0},
            {"timeout_ms": 0},
            {"retry_budget_s": -1},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                DreamPolicy(**kwargs)


class SessionTests(unittest.TestCase):
    def test_first_reset_opens_the_policy_with_the_yam_config(self):
        client = FakeClient()
        opened(client, obs_window=3, control_hz=30.0, head="eef")
        self.assertEqual(
            client.config,
            {
                "checkpoint": "dreamlabs/dream-1-beta",
                "timeout_ms": 15000,
                "strict": True,
                "action_hz": 30.0,
                "obs_window": 3,
                "ik": {"arm": "yam", "head": "eef"},
                "embodiment": yam.EMBODIMENT,
            },
        )
        self.assertEqual(client.resets, 0)

    def test_later_resets_end_the_episode_and_forget_the_last_command(self):
        client = FakeClient([reply(), reply()])
        policy = opened(client)
        policy.act(observation())
        policy.reset(SCENE)
        policy.act(observation())
        self.assertEqual(client.resets, 1)
        self.assertIsNone(client.asked[1])

    def test_a_refused_config_closes_the_client(self):
        client = FakeClient()

        def refuse(config):
            raise dream.DreamConfigError("ik needs dream_ik")

        client.policy = refuse
        policy = DreamPolicy(client_factory=lambda: client)
        with self.assertRaises(dream.DreamConfigError):
            policy.reset(SCENE)
        self.assertTrue(client.closed)
        with self.assertRaises(RuntimeError):
            policy.act(observation())

    def test_close_closes_once(self):
        client = FakeClient()
        with opened(client) as policy:
            pass
        self.assertTrue(client.closed)
        policy.close()

    def test_act_before_reset_is_refused(self):
        with self.assertRaises(RuntimeError):
            DreamPolicy(client_factory=FakeClient).act(observation())


class ActTests(unittest.TestCase):
    def test_sends_three_frames_the_split_state_and_the_prompt(self):
        client = FakeClient([reply()])
        opened(client).act(observation())
        ((frames, sent),) = client.ticks
        self.assertEqual(set(frames), {"video.overhead", "video.wrist_l", "video.wrist_r"})
        for key, shade in (("video.overhead", 40), ("video.wrist_l", 80), ("video.wrist_r", 120)):
            picture = Image.open(io.BytesIO(base64.b64decode(frames[key])))
            self.assertEqual((picture.format, picture.size), ("JPEG", yam.FRAME_SIZE))
            self.assertAlmostEqual(np.asarray(picture).mean(), shade, delta=2)
        state = np.arange(14) / 10
        np.testing.assert_allclose(sent["state.left_arm"], state[0:6], rtol=1e-6)
        np.testing.assert_allclose(sent["state.left_gripper"], state[6:7], rtol=1e-6)
        np.testing.assert_allclose(sent["state.right_arm"], state[7:13], rtol=1e-6)
        np.testing.assert_allclose(sent["state.right_gripper"], state[13:14], rtol=1e-6)
        self.assertEqual(sent["state.left_arm"].dtype, np.float32)
        self.assertEqual(sent["prompt"], "fold the towel")

    def test_an_observation_instruction_overrides_the_scene(self):
        client = FakeClient([reply()])
        opened(client).act(observation(instruction="now hang it up"))
        self.assertEqual(client.ticks[0][1]["prompt"], "now hang it up")

    def test_camera_names_map_in_role_order(self):
        client = FakeClient([reply()])
        policy = opened(client, cameras="right_cam, top_cam, left_cam")
        policy.act(observation())
        frames = client.ticks[0][0]
        overhead = Image.open(io.BytesIO(base64.b64decode(frames["video.overhead"])))
        self.assertAlmostEqual(np.asarray(overhead).mean(), 120, delta=2)

    def test_packs_the_reply_left_then_right(self):
        client = FakeClient([reply(rows=5, left=0.1, right=0.2)])
        chunk = opened(client).act(observation())
        self.assertEqual(len(chunk), 5)
        np.testing.assert_allclose(chunk.actions[0].data, [0.1] * 7 + [0.2] * 7)
        self.assertEqual(chunk.control_hz, 20.0)
        self.assertIsNotNone(chunk.inference_latency_s)
        self.assertEqual(chunk.meta["dream_session"], "ses_fake")

    def test_the_next_call_is_anchored_on_the_last_row_played(self):
        last = reply(rows=4)
        last["q_l_chunk"][-1] = [1, 2, 3, 4, 5, 6, 0.5]
        client = FakeClient([last, reply()])
        policy = opened(client)
        policy.act(observation())
        policy.act(observation())
        self.assertIsNone(client.asked[0])
        np.testing.assert_allclose(client.asked[1]["q_l_chunk"], [1, 2, 3, 4, 5, 6, 0.5])
        np.testing.assert_allclose(client.asked[1]["q_r_chunk"], [0.2] * 7)

    def test_a_missing_camera_or_state_is_named(self):
        policy = opened(FakeClient([reply()]))
        obs = observation()
        no_camera = type(obs)(images={"top_cam": obs.images["top_cam"]}, state=obs.state)
        with self.assertRaisesRegex(ValueError, "left_cam"):
            policy.act(no_camera)
        with self.assertRaisesRegex(ValueError, "joint_pos"):
            policy.act(type(obs)(images=obs.images, state={}))

    def test_malformed_replies_are_refused(self):
        bad = (
            {"q_r_chunk": reply()["q_r_chunk"]},
            {**reply(), "q_l_chunk": np.zeros((48, 6)).tolist()},
            {**reply(), "q_l_chunk": np.zeros((0, 7)).tolist()},
            {**reply(rows=3), "q_r_chunk": np.zeros((4, 7)).tolist()},
            {**reply(), "q_l_chunk": np.full((48, 7), np.nan).tolist()},
        )
        for answer in bad:
            with self.subTest(answer=sorted(answer)), self.assertRaises(ValueError):
                opened(FakeClient([answer])).act(observation())


class RetryTests(unittest.TestCase):
    def policy(self, client, budget=60.0):
        clock = FakeClock()
        policy = DreamPolicy(
            client_factory=lambda: client,
            retry_budget_s=budget,
            clock=clock,
            sleep=clock.sleep,
        )
        policy.reset(SCENE)
        return policy, clock

    def test_a_rate_limit_waits_and_asks_again(self):
        limited = dream.DreamServerError(
            "too fast", "DA-2000", "rate_limited", retryable=True, retry_after_s=0.7
        )
        client = FakeClient([limited, reply()])
        policy, clock = self.policy(client)
        self.assertEqual(len(policy.act(observation())), 48)
        self.assertEqual(clock.sleeps, [0.7])

    def test_an_expired_session_or_timeout_asks_again_at_once(self):
        expired = dream.DreamSessionExpired("replaced", "DA-0001", "expired")
        client = FakeClient([expired, dream.DreamTimeout("slow"), reply()])
        policy, clock = self.policy(client)
        policy.act(observation())
        self.assertEqual(clock.sleeps, [0.0, 0.0])

    def test_a_refusal_that_is_not_retryable_raises_at_once(self):
        refused = dream.DreamAuthError("unknown key", "DA-1000", "unknown_key")
        policy, clock = self.policy(FakeClient([refused]))
        with self.assertRaises(dream.DreamAuthError):
            policy.act(observation())
        self.assertEqual(clock.sleeps, [])

    def test_a_warming_model_raises_once_the_budget_is_spent(self):
        warming = [
            dream.DreamServerError(
                "starting", "DA-2000", "pool_warming", retryable=True, retry_after_s=30.0
            )
            for _ in range(3)
        ]
        policy, clock = self.policy(FakeClient(warming), budget=60.0)
        with self.assertRaisesRegex(dream.DreamServerError, "starting"):
            policy.act(observation())
        self.assertEqual(clock.sleeps, [30.0, 30.0])
