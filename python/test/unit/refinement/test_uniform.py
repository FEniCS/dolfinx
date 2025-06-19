from mpi4py import MPI

import pytest

import dolfinx
from dolfinx.mesh import CellType, Mesh, create_unit_cube, create_unit_square


@pytest.mark.parametrize("ctype", [CellType.hexahedron, CellType.tetrahedron, CellType.prism])
def test_uniform_refinement_3d(ctype):
    mesh = create_unit_cube(MPI.COMM_WORLD, 3, 2, 1, cell_type=ctype)

    ncells0 = mesh.topology.index_map(3).size_local
    mesh.topology.create_entities(1)
    mesh.topology.create_entities(2)

    m2 = Mesh(dolfinx.cpp.refinement.uniform_refine(mesh._cpp_object), None)
    ncells1 = m2.topology.index_map(3).size_local

    assert mesh.comm.allreduce(ncells0) * 8 == mesh.comm.allreduce(ncells1)


@pytest.mark.parametrize("ctype", [CellType.triangle, CellType.quadrilateral])
def test_uniform_refinement_2d(ctype):
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 11, cell_type=ctype)

    ncells0 = mesh.topology.index_map(2).size_local
    mesh.topology.create_entities(1)

    m2 = Mesh(dolfinx.cpp.refinement.uniform_refine(mesh._cpp_object), None)
    ncells1 = m2.topology.index_map(2).size_local

    assert mesh.comm.allreduce(ncells0) * 4 == mesh.comm.allreduce(ncells1)
