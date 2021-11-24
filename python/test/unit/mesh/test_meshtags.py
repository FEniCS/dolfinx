# Copyright (C) 2019-2021 Michel Habera
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


import numpy as np
import pytest
from dolfinx.generation import UnitCubeMesh
from dolfinx.mesh import CellType, create_meshtags, locate_entities
from mpi4py import MPI


@pytest.mark.parametrize("cell_type", [CellType.tetrahedron, CellType.hexahedron])
def test_create(cell_type):
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 6, 6, 6, cell_type)
    marked_lines = locate_entities(mesh, 1, lambda x: np.isclose(x[1], 0.5))
    f_v = mesh.topology.connectivity(1, 0).array.reshape(-1, 2)
    entities = f_v[marked_lines]
    values = np.full(marked_lines.shape[0], 2, dtype=np.int32)
    mt = create_meshtags(mesh, 1, entities, values)
    assert mt.indices.shape == marked_lines.shape
