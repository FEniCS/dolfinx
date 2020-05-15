from dolfinx.generation import UnitSquareMesh
from dolfinx.mesh import locate_entities_boundary
import dolfinx.cpp

from mpi4py import MPI
import numpy


def test_create():
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 4, 4)
    mesh.topology.create_connectivity_all()

    bottom_facets = locate_entities_boundary(mesh, 1, lambda x: numpy.isclose(x[1], 0.0))
    print(MPI.COMM_WORLD.rank, "bottom", bottom_facets)

    f_v = mesh.topology.connectivity(1, 0).array().reshape(-1, 2)

    # TODO: make global_indices return numpy array
    global_idx = numpy.asarray(mesh.topology.index_map(0).global_indices(True))
    global_idx_f = numpy.asarray(mesh.topology.index_map(1).global_indices(True))

    print(MPI.COMM_WORLD.rank, "global idx vertex", global_idx)
    print(MPI.COMM_WORLD.rank, "global idx facet", global_idx_f)
    print(MPI.COMM_WORLD.rank, "global idx entities vtx", global_idx[f_v[bottom_facets]])

    entities = dolfinx.cpp.graph.AdjacencyList64(global_idx[f_v[bottom_facets]])
    print(MPI.COMM_WORLD.rank, "entities")
    values = numpy.full(bottom_facets.shape[0], 2, dtype=numpy.int32)

    mt = dolfinx.cpp.mesh.create_meshtags(mesh, dolfinx.cpp.mesh.CellType.interval, entities, values)

    print(MPI.COMM_WORLD.rank, "mt.indices", mt.indices)



    assert bottom_facets.shape == mt.indices.shape