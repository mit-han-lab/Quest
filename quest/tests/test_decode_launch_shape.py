"""Cover both decode launch shapes on any GPU.

`begin_forward` chooses between the single-block decode kernel and the split
(partition-kv) one from the device occupancy, so a given GPU only ever reaches
one of them and the other stays untested there. Setting
QUEST_DECODE_LAUNCH_SHAPE to `single` or `split` forces the choice. It is read
in the process that runs the kernels, so each shape is covered here by
re-running the decode suites in a subprocess with the variable set. A value the
run cannot honour raises instead of falling back to occupancy, which is what
makes a green run below evidence that the requested shape really ran.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent
DECODE_SUITES = ["test_decode_attention.py", "test_approx_attention.py"]


def _run_suites(shape, suites, extra_args=()):
    env = dict(os.environ)
    env["QUEST_DECODE_LAUNCH_SHAPE"] = shape
    cmd = [sys.executable, "-m", "pytest", "-q", *extra_args]
    cmd += [str(TESTS_DIR / name) for name in suites]
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)


@pytest.mark.parametrize("shape", ["single", "split"])
def test_decode_suites_pass_with_forced_launch_shape(shape):
    result = _run_suites(shape, DECODE_SUITES)
    assert result.returncode == 0, result.stdout + result.stderr


def test_unrecognized_launch_shape_is_rejected():
    result = _run_suites("not-a-shape", DECODE_SUITES[:1], extra_args=("-x",))
    assert result.returncode != 0, result.stdout + result.stderr
    assert "QUEST_DECODE_LAUNCH_SHAPE must be" in result.stdout + result.stderr
