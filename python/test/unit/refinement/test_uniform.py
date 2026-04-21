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

    m2 = dolfinx.mesh.uniform_refine(mesh)
    ncells1 = m2.topology.index_map(3).size_local

    assert mesh.comm.allreduce(ncells0) * 8 == mesh.comm.allreduce(ncells1)


@pytest.mark.parametrize("ctype", [CellType.triangle, CellType.quadrilateral])
def test_uniform_refinement_2d(ctype):
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 11, cell_type=ctype)

    ncells0 = mesh.topology.index_map(2).size_local
    mesh.topology.create_entities(1)

    m2 = dolfinx.mesh.uniform_refine(mesh)
    ncells1 = m2.topology.index_map(2).size_local

    assert mesh.comm.allreduce(ncells0) * 4 == mesh.comm.allreduce(ncells1)


def test_uniform_refine_mixed_mesh(mixed_topology_mesh):
    mesh = Mesh(mixed_topology_mesh, None)

    ct = mesh.topology.entity_types[3]
    ncells0 = {ct[j]: mesh.topology.index_maps(3)[j].size_local for j in range(4)}
    print(ncells0)
    mesh.topology.create_entities(1)
    mesh.topology.create_entities(2)

    m2 = Mesh(dolfinx.cpp.refinement.uniform_refine(mesh._cpp_object, None), None)
    ncells1 = {ct[j]: m2.topology.index_maps(3)[j].size_local for j in range(4)}

    comm = mesh.comm
    assert comm.allreduce(ncells0[CellType.hexahedron]) * 8 == comm.allreduce(
        ncells1[CellType.hexahedron]
    )
    assert comm.allreduce(ncells0[CellType.prism]) * 8 == comm.allreduce(ncells1[CellType.prism])
    assert comm.allreduce(ncells0[CellType.pyramid]) * 5 == comm.allreduce(
        ncells1[CellType.pyramid]
    )
    assert comm.allreduce(ncells0[CellType.tetrahedron]) * 8 + comm.allreduce(
        ncells0[CellType.pyramid]
    ) * 4 == comm.allreduce(ncells1[CellType.tetrahedron])
