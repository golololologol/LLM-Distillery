import numpy as np
from numpy.testing import assert_allclose
from classes.data_manager import _compress_distributions, _decompress_distributions


def _roundtrip(data):
    packed, offsets = _compress_distributions(data)
    return _decompress_distributions(packed, offsets, data.shape[0])


def test_roundtrip_random():
    data = np.random.dirichlet(np.ones(256), size=100).astype(np.float32)
    result = _roundtrip(data)
    assert_allclose(result, data, atol=1e-3)


def test_roundtrip_sparse():
    data = np.zeros((50, 256), dtype=np.float32)
    for i in range(50):
        idx = np.random.choice(256, 5, replace=False)
        data[i, idx] = np.random.dirichlet(np.ones(5))
    result = _roundtrip(data)
    assert_allclose(result, data, atol=1e-3)


def test_roundtrip_dense():
    data = np.random.dirichlet(np.ones(256), size=30).astype(np.float32)
    result = _roundtrip(data)
    assert_allclose(result, data, atol=1e-3)


def test_roundtrip_single_nonzero():
    data = np.zeros((20, 256), dtype=np.float32)
    for i in range(20):
        data[i, np.random.randint(256)] = 1.0
    result = _roundtrip(data)
    assert_allclose(result, data, atol=1e-3)


def test_roundtrip_all_zero_row():
    data = np.zeros((5, 256), dtype=np.float32)
    result = _roundtrip(data)
    assert_allclose(result, data, atol=1e-8)


def test_mass_preservation():
    data = np.random.dirichlet(np.ones(256), size=50).astype(np.float32)
    result = _roundtrip(data)
    assert_allclose(result.sum(axis=1), data.sum(axis=1), atol=1e-2)


def test_offsets_monotonic():
    data = np.random.dirichlet(np.ones(256), size=40).astype(np.float32)
    _, offsets = _compress_distributions(data)
    assert np.all(offsets[1:] >= offsets[:-1])


def test_single_row():
    data = np.random.dirichlet(np.ones(256), size=1).astype(np.float32)
    result = _roundtrip(data)
    assert_allclose(result, data, atol=1e-3)


def test_mixed_sparsity():
    data = np.zeros((10, 256), dtype=np.float32)
    # sparse rows
    for i in range(3):
        idx = np.random.choice(256, 3, replace=False)
        data[i, idx] = np.random.dirichlet(np.ones(3))
    # dense rows
    for i in range(3, 7):
        data[i] = np.random.dirichlet(np.ones(256))
    # zero rows (7-9 already zero)
    result = _roundtrip(data)
    assert_allclose(result, data, atol=1e-3)
