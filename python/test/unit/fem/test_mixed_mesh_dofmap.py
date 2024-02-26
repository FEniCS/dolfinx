from mpi4py import MPI

import numpy as np

import basix
import dolfinx.cpp as _cpp
from dolfinx import jit
from dolfinx.cpp.mesh import Mesh_float64, create_geometry, create_topology
from dolfinx.fem import coordinate_element
from dolfinx.fem.dofmap import DofMap
from dolfinx.log import LogLevel, set_log_level
from dolfinx.mesh import CellType


def create_element_dofmap(mesh, cell_types, degree):
    cpp_elements = []
    dofmaps = []
    for cell_type in cell_types:
        ufl_e = basix.ufl.element("P", cell_type, degree)
        form_compiler_options = {"scalar_type": np.float64}
        (ufcx_element, ufcx_dofmap), module, code = jit.ffcx_jit(
            mesh.comm, ufl_e, form_compiler_options=form_compiler_options
        )
        ffi = module.ffi
        cpp_elements += [
            _cpp.fem.FiniteElement_float64(ffi.cast("uintptr_t", ffi.addressof(ufcx_element)))
        ]
        dofmaps += [ufcx_dofmap]

    cpp_dofmaps = _cpp.fem.create_dofmaps(
        mesh.comm, [ffi.cast("uintptr_t", ffi.addressof(dm)) for dm in dofmaps], mesh.topology
    )

    return (cpp_elements, cpp_dofmaps)


def test_el_dm():
    rank = MPI.COMM_WORLD.Get_rank()

    # Two triangles and one quadrilateral
    tri = [0, 1, 4, 0, 3, 4]
    quad = [1, 4, 2, 5]
    # cells with global indexing
    cells = [[t + 3 * rank for t in tri], [q + 3 * rank for q in quad]]
    orig_index = [[3 * rank, 1 + 3 * rank], [2 + 3 * rank]]
    # No ghosting
    ghost_owners = [[], []]
    # All vertices are on boundary
    boundary_vertices = [3 * rank + i for i in range(6)]

    topology = create_topology(
        MPI.COMM_WORLD,
        [CellType.triangle, CellType.quadrilateral],
        cells,
        orig_index,
        ghost_owners,
        boundary_vertices,
    )
    # Create dofmaps for Geometry
    tri = coordinate_element(CellType.triangle, 1)
    quad = coordinate_element(CellType.quadrilateral, 1)
    nodes = np.arange(6, dtype=np.int64) + 3 * rank
    xdofs = np.array([0, 1, 4, 0, 3, 4, 1, 4, 2, 5], dtype=np.int64) + 3 * rank
    x = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], dtype=np.float64
    )
    x[:, 1] += 1.0 * rank

    set_log_level(LogLevel.INFO)
    geom = create_geometry(
        topology, [tri._cpp_object, quad._cpp_object], nodes, xdofs, x.flatten(), 2
    )
    mesh = Mesh_float64(MPI.COMM_WORLD, topology, geom)

    assert mesh.geometry.x.shape == (6, 3)

    # Second order dofmap on mixed mesh
    el, dm = create_element_dofmap(mesh, [basix.CellType.triangle, basix.CellType.quadrilateral], 2)

    assert len(el) == 2
    assert el[0].basix_element.cell_type.name == "triangle"
    assert el[1].basix_element.cell_type.name == "quadrilateral"

    assert len(dm) == 2
    q0 = DofMap(dm[0])
    q1 = DofMap(dm[1])
    assert q0.index_map.size_local == q1.index_map.size_local
    # Triangles
    print(q0.list)
    assert q0.list.shape == (2, 6)
    assert len(q0.dof_layout.entity_dofs(2, 0)) == 0
    # Quadrilaterals
    print(q1.list)
    assert q1.list.shape == (1, 9)
    assert len(q1.dof_layout.entity_dofs(2, 0)) == 1


def test_el_dm_prism():
    # Prism mesh
    cells = [[0, 1, 2, 3, 4, 5]]
    # cells with global indexing
    orig_index = [[0, 1, 2, 3, 4, 5]]
    # No ghosting
    ghost_owners = [[]]
    # All vertices are on boundary
    boundary_vertices = [0, 1, 2, 3, 4, 5]

    topology = create_topology(
        MPI.COMM_SELF, [CellType.prism], cells, orig_index, ghost_owners, boundary_vertices
    )
    topology.create_entities(2)

    # Create dofmaps for Geometry
    prism = coordinate_element(CellType.prism, 1)
    nodes = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    xdofs = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    x = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    set_log_level(LogLevel.INFO)
    geom = create_geometry(topology, [prism._cpp_object], nodes, xdofs, x.flatten(), 3)
    mesh = Mesh_float64(MPI.COMM_WORLD, topology, geom)

    el, dm = create_element_dofmap(mesh, [basix.CellType.prism], 2)
    print()
    assert len(el) == 1
    assert len(dm) == 1
    q = DofMap(dm[0])
    assert q.index_map.size_local == 18
    print(q.list)
    facet_dofs = []
    for j in range(5):
        facet_dofs += q.dof_layout.entity_dofs(2, j)
    assert len(facet_dofs) == 3
