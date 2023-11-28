# Copyright (C) 2022 Michal Habera and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

from dolfinx.graph import adjacencylist
from dolfinx.mesh import CellType, create_unit_cube, locate_entities, meshtags_from_entities
from ufl import Measure

celltypes_3D = [CellType.tetrahedron, CellType.hexahedron]


@pytest.mark.parametrize("cell_type", celltypes_3D)
def test_create(cell_type):
    comm = MPI.COMM_WORLD
    mesh = create_unit_cube(comm, 6, 6, 6, cell_type)

    marked_lines = locate_entities(mesh, 1, lambda x: np.isclose(x[1], 0.5))
    f_v = mesh.topology.connectivity(1, 0).array.reshape(-1, 2)

    entities = adjacencylist(f_v[marked_lines])
    values = np.full(marked_lines.shape[0], 2, dtype=np.int32)
    mt = meshtags_from_entities(mesh, 1, entities, values)

    assert hasattr(mt, "ufl_id")
    assert mt.indices.shape == marked_lines.shape
    assert mt.values.dtype == np.int32
    assert mt.values.shape[0] == entities.num_nodes


def test_ufl_id():
    """Test that UFL can process MeshTags (tests ufl_id attribute)"""
    comm = MPI.COMM_WORLD
    msh = create_unit_cube(comm, 6, 6, 6)
    tdim = msh.topology.dim
    marked_facets = locate_entities(msh, tdim - 1, lambda x: np.isclose(x[1], 1))
    f_v = msh.topology.connectivity(tdim - 1, 0).array.reshape(-1, 3)

    entities = adjacencylist(f_v[marked_facets])
    values = np.full(marked_facets.shape[0], 2, dtype=np.int32)
    ft = meshtags_from_entities(msh, tdim - 1, entities, values)
    ds = Measure("ds", domain=msh, subdomain_data=ft, subdomain_id=(2, 3))
    a = 1 * ds
    assert isinstance(a.subdomain_data(), dict)
