import numpy as np
import pytest

from dolfinx.graph import create_adjacencylist
from dolfinx.mesh import (CellType, create_meshtags, create_unit_cube_mesh,
                          locate_entities)

from mpi4py import MPI

celltypes_3D = [CellType.tetrahedron, CellType.hexahedron]


@pytest.mark.parametrize("cell_type", celltypes_3D)
def test_create(cell_type):
    comm = MPI.COMM_WORLD

    mesh = create_unit_cube_mesh(comm, 6, 6, 6, cell_type)

    marked_lines = locate_entities(mesh, 1, lambda x: np.isclose(x[1], 0.5))
    f_v = mesh.topology.connectivity(1, 0).array.reshape(-1, 2)

    entities = create_adjacencylist(f_v[marked_lines])
    values = np.full(marked_lines.shape[0], 2, dtype=np.int32)

    mt = create_meshtags(mesh, 1, entities, values)
    assert mt.indices.shape == marked_lines.shape
