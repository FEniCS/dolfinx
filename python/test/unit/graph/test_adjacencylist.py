# Copyright (C) 2021-2026 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
import pytest

from dolfinx.graph import adjacencylist


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_create_adj2d(dtype):
    data = np.zeros([2, 4], dtype=dtype)
    adj = adjacencylist(data)
    num_nodes, num_links = data.shape[0], data.shape[1]
    assert np.array_equal(
        adj.offsets, np.arange(0, num_nodes * num_links + num_links, num_links, dtype=np.int32)
    )

    data = np.arange(20, dtype=dtype)
    offsets = np.array([0, 5, 15, 20], dtype=np.int32)
    adj = adjacencylist(data, offsets)
    assert adj.num_nodes == 3
    assert len(adj.links(0)) == 5
    assert len(adj.links(1)) == 10
    assert len(adj.links(2)) == 5


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_equality_is_unhashable(dtype):
    """Structurally equal adjacency lists must not have identity hashes."""
    data = np.array([[1, 2], [0, 2]], dtype=dtype)
    adj0 = adjacencylist(data)
    adj1 = adjacencylist(data.copy())

    assert adj0 == adj1
    with pytest.raises(TypeError):
        hash(adj0)

    # The wrapped C++ class also compares structurally, so it must be
    # unhashable too
    assert adj0._cpp_object == adj1._cpp_object
    with pytest.raises(TypeError):
        hash(adj0._cpp_object)
