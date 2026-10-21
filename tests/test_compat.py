"""The ``dream`` policy against Inspect's registry and the YAM embodiment's declaration."""

import importlib.util
import unittest

from inspect_robots.compat import check_compatibility
from inspect_robots.conformance import missing_runtime_requirements
from inspect_robots.registry import registered

from fakes import FakeYam
from inspect_robots_dream import DreamPolicy, yam


class CompatTests(unittest.TestCase):
    def test_registry_names_it_dream(self):
        factory = registered("policy")["dream"]
        self.assertIs(factory, DreamPolicy)
        self.assertEqual(factory().info.name, "dream")

    def test_pairs_with_the_yam_declaration(self):
        report = check_compatibility(DreamPolicy(), FakeYam())
        self.assertEqual(report.issues, [])

    def test_a_slower_yam_is_a_control_rate_warning(self):
        report = check_compatibility(DreamPolicy(control_hz=30.0), FakeYam())
        self.assertTrue(report.ok)
        self.assertEqual([issue.code for issue in report.warnings], ["control_rate"])

    def test_runtime_requirements_name_their_install(self):
        missing = missing_runtime_requirements(DreamPolicy)
        self.assertNotIn("dream", missing)
        if importlib.util.find_spec("dream_ik") is None:
            self.assertEqual(
                missing["dream_ik"], "python -m pip install 'inspect-robots-dream[ik]'"
            )

    @unittest.skipUnless(
        importlib.util.find_spec("inspect_robots_yam"), "inspect-robots-yam is not installed"
    )
    def test_packing_matches_inspect_robots_yam(self):
        from inspect_robots_yam import packing

        self.assertEqual(packing.DIM_LABELS, yam.DIM_LABELS)
        self.assertEqual((packing.LEFT, packing.RIGHT), (yam.LEFT, yam.RIGHT))

    @unittest.skipUnless(
        importlib.util.find_spec("inspect_robots_yam"), "inspect-robots-yam is not installed"
    )
    def test_pairs_with_the_real_yam_arms_at_twenty_hz(self):
        robot = registered("embodiment")["yam_arms"](control_hz=20.0)
        self.assertEqual(check_compatibility(DreamPolicy(), robot).issues, [])
