# Copyright (C) 2021 Garth N. Wells
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
