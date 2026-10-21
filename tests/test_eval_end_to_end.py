"""End-to-end: a full eval() of the dream policy on a fake YAM, with a fake Dream client."""

import tempfile
import unittest

import numpy as np
from inspect_robots import Scene, Task
from inspect_robots import eval as rl_eval
from inspect_robots.scorer import success_at_end

from fakes import FakeClient, FakeYam, reply
from inspect_robots_dream import DreamPolicy


class EndToEndTests(unittest.TestCase):
    def test_eval_plays_every_row_and_starts_each_trial_afresh(self):
        client = FakeClient([reply(rows=4, left=0.1), reply(rows=4, left=0.3)] * 2)
        task = Task(
            name="towel",
            scenes=[Scene(id="towel", instruction="fold the towel")],
            scorer=success_at_end(),
            max_steps=8,
            epochs=2,
        )
        robot = FakeYam()
        with tempfile.TemporaryDirectory() as log_dir:
            (log,) = rl_eval(
                task, DreamPolicy(client_factory=lambda: client), robot, log_dir=log_dir
            )
        self.assertEqual(log.status, "success")
        self.assertEqual(len(robot.targets), 16)
        np.testing.assert_allclose(robot.targets[0][:7], 0.1)
        np.testing.assert_allclose(robot.targets[4][:7], 0.3)
        self.assertEqual(client.resets, 1)
        self.assertEqual(len(client.ticks), 4)
        self.assertEqual([c is None for c in client.asked], [True, False, True, False])
