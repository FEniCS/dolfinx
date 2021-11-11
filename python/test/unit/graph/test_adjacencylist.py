
import numpy as np
import pytest
from dolfinx import graph


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_create_adj2d(dtype):
    data = np.zeros([13, 4], dtype=dtype)
    adj = graph.create_adjacencylist(data)
    assert np.array_equal(adj.offsets, np.arange(0, data.shape[0] + data.shape[1], data.shape[1], dtype=np.int32))
