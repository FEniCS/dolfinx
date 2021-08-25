# TODO When cell / facet spaces are working properly, improve these tests.
# They are currently very simplistic.
import dolfinx
from mpi4py import MPI


def test_mesh_view_cell_space():
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 4, 2)

    dim = mesh.topology.dim
    entities = dolfinx.mesh.locate_entities(mesh, dim, lambda x: x[0] <= 0.5)
    mv_cpp = dolfinx.cpp.mesh.MeshView(mesh, dim, entities)

    V = dolfinx.FunctionSpace(mv_cpp, ("Lagrange", 1))

    assert V.dofmap.list.num_nodes == len(entities)


def test_mesh_view_facet_space():
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 4, 2)

    dim = mesh.topology.dim - 1
    entities = dolfinx.mesh.locate_entities_boundary(mesh, dim,
                                                     lambda x: x[0] <= 0.5)
    mv_cpp = dolfinx.cpp.mesh.MeshView(mesh, dim, entities)

    V = dolfinx.FunctionSpace(mv_cpp, ("Lagrange", 1))

    assert V.dofmap.list.num_nodes == len(entities)
