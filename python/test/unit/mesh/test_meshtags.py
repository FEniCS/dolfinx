from dolfinx.generation import UnitCubeMesh, UnitSquareMesh
from dolfinx.mesh import locate_entities
from dolfinx.cpp.mesh import CellType
from dolfinx.io import XDMFFile
from dolfinx import cpp
import pytest

from mpi4py import MPI
import numpy

celltypes_3D = [CellType.tetrahedron, CellType.hexahedron]


@pytest.mark.parametrize("cell_type", celltypes_3D)
def test_create(cell_type):
    comm = MPI.COMM_WORLD

    mesh = UnitCubeMesh(comm, 6, 6, 6, cell_type)
    mesh.topology.create_connectivity_all()

    marked_lines = locate_entities(mesh, 1, lambda x: numpy.isclose(x[1], 0.5))
    f_v = mesh.topology.connectivity(1, 0).array().reshape(-1, 2)

    entities = cpp.graph.AdjacencyList_int32(f_v[marked_lines])
    values = numpy.full(marked_lines.shape[0], 2, dtype=numpy.int32)

    mt = cpp.mesh.create_meshtags(mesh, cpp.mesh.CellType.interval, entities, values)
    assert mt.indices.shape == marked_lines.shape
