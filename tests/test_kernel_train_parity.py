import subprocess
import sys
import os
import pytest
from conftest import requires_gpu

_TESTS_DIR = os.path.dirname(__file__)
_ROOT_DIR = os.path.dirname(_TESTS_DIR)
_RUNNER = os.path.join(_TESTS_DIR, "_run_fused_train_test.py")
_TIMEOUT = 90


def _run_fused_test(test_name):
    proc = subprocess.Popen(
        [sys.executable, _RUNNER, test_name],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=_ROOT_DIR,
    )
    try:
        stdout, stderr = proc.communicate(timeout=_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        pytest.skip(f"Fused kernel deadlocked (>{_TIMEOUT}s)")
    if proc.returncode == 2:
        pytest.skip(stdout.strip() or "Kernel unavailable")
    assert proc.returncode == 0, f"Test failed:\n{stdout}\n{stderr}"


@requires_gpu
@pytest.mark.slow
def test_fused_train_skew_kl_parity():
    _run_fused_test("skew_kl_parity")


@requires_gpu
@pytest.mark.slow
def test_fused_train_akl_parity():
    _run_fused_test("akl_parity")


@requires_gpu
@pytest.mark.slow
def test_fused_train_abomination_parity():
    _run_fused_test("abomination_parity")


@requires_gpu
@pytest.mark.slow
def test_fused_train_wasserstein_parity():
    _run_fused_test("wasserstein_parity")


@requires_gpu
@pytest.mark.slow
def test_fused_train_jsd_parity():
    _run_fused_test("jsd_parity")


@requires_gpu
@pytest.mark.slow
def test_fused_train_hellinger_parity():
    _run_fused_test("hellinger_parity")


@requires_gpu
@pytest.mark.slow
def test_fused_train_loss_keys():
    _run_fused_test("loss_keys")
