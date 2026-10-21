"""Inspect Robots policy adapter for Dream Labs' Dream-1, called through the Dream API.

Registers one Inspect Robots component via entry point:

* policy ``dream`` -- :class:`~inspect_robots_dream.policy.DreamPolicy`

so ``inspect-robots run --policy dream --embodiment yam_arms --instruction ...`` works once
the package and inspect-robots-yam are installed and a Dream API key is saved.
"""

from __future__ import annotations

from inspect_robots_dream.policy import DreamPolicy

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("inspect-robots-dream")
except PackageNotFoundError:  # Only in a source tree that is not installed.
    __version__ = "0.0.0+unknown"

__all__ = ["DreamPolicy", "__version__"]
