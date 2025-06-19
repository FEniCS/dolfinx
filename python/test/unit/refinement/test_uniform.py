from mpi4py import MPI

import pytest

import dolfinx
from dolfinx.mesh import CellType, Mesh, create_unit_cube


@pytest.mark.parametrize("ctype", [CellType.hexahedron, CellType.tetrahedron, CellType.prism])
def test_uniform_refinement(ctype):
    mesh = create_unit_cube(MPI.COMM_WORLD, 3, 2, 1, cell_type=ctype)

    ncells0 = mesh.topology.index_map(3).size_local
    mesh.topology.create_entities(1)
    mesh.topology.create_entities(2)

    m2 = Mesh(dolfinx.cpp.refinement.uniform_refine(mesh._cpp_object), None)
    ncells1 = m2.topology.index_map(3).size_local

    assert ncells0 * 8 == ncells1
