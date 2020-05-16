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
@pytest.mark.parametrize("mode", [cpp.mesh.GhostMode.none,
                                  pytest.param(cpp.mesh.GhostMode.shared_facet,
                                               marks=pytest.mark.xfail(condition=MPI.COMM_WORLD.size == 1,
                                                                       reason="Shared ghost modes fail in serial"))])
def test_create(cell_type, mode):
    comm = MPI.COMM_WORLD

    mesh = UnitCubeMesh(comm, 6, 6, 6, cell_type, ghost_mode=mode)
    mesh.topology.create_connectivity_all()

    marked_lines = locate_entities(mesh, 1, lambda x: numpy.isclose(x[1], 0.5))
    f_v = mesh.topology.connectivity(1, 0).array().reshape(-1, 2)

    # TODO: make global_indices return numpy array
    global_idx = numpy.asarray(mesh.topology.index_map(0).global_indices(True))

    entities = cpp.graph.AdjacencyList64(global_idx[f_v[marked_lines]])
    values = numpy.full(marked_lines.shape[0], 2, dtype=numpy.int32)

    mt = cpp.mesh.create_meshtags(mesh, cpp.mesh.CellType.interval, entities, values)

    lines_local = comm.allreduce((mt.indices < mesh.topology.index_map(1).size_local).sum(), op=MPI.SUM)
    lines_local_marked = comm.allreduce((marked_lines < mesh.topology.index_map(1).size_local).sum(), op=MPI.SUM)

    assert lines_local == lines_local_marked
