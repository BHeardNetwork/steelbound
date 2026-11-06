"""Release-gate checks for the v0.2 review: the shipped version must be
consistent everywhere it is declared, and the changelog must carry a
section for that version with nothing left pending under [Unreleased].
These run as ordinary unit tests so a release can never be tagged with
a version/changelog mismatch.
"""
import re
import unittest
from pathlib import Path

from steelbound import __version__

ROOT = Path(__file__).resolve().parent.parent


def _section(text, heading):
    """Return the body of the markdown section starting with heading."""
    start = text.index(heading)
    rest = text[start + len(heading):]
    m = re.search(r"\n## ", rest)
    return rest[:m.start()] if m else rest


class TestRelease(unittest.TestCase):
    def test_version_matches_pyproject(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        declared = re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1)
        self.assertEqual(__version__, declared)

    def test_changelog_has_section_for_version(self):
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIn(f"## [{__version__}]", changelog)

    def test_unreleased_section_is_empty(self):
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        unreleased = _section(changelog, "## [Unreleased]")
        bullets = [ln for ln in unreleased.splitlines()
                   if ln.strip().startswith("- ") and "(none yet)" not in ln]
        self.assertEqual(bullets, [])


if __name__ == "__main__":
    unittest.main()
