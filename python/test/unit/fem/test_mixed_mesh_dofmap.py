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


def create_element_dofmap(mesh):
    cpp_elements = []
    dofmaps = []
    for cell_type in [basix.CellType.triangle, basix.CellType.quadrilateral]:
        ufl_e = basix.ufl.element("P", cell_type, 2)
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
    nodes = [3 * rank + i for i in range(6)]
    xdofs = np.array([0, 1, 2, 1, 2, 3, 2, 3, 4, 5], dtype=int) + 3 * rank
    x = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [2.0, 0.0]], dtype=np.float64
    )
    x[:, 1] += 1.0 * rank

    set_log_level(LogLevel.INFO)
    geom = create_geometry(
        topology, [tri._cpp_object, quad._cpp_object], nodes, xdofs, x.flatten(), 2
    )
    mesh = Mesh_float64(MPI.COMM_WORLD, topology, geom)

    el, dm = create_element_dofmap(mesh)
    print()
    for e, d in zip(el, dm):
        print(e.basix_element.cell_type.name)
        q = DofMap(d)
        print(q.index_map.size_local)
        print(q.list)
        print(q.dof_layout.entity_dofs(2, 0))


def test_el_dm_prism():
    # Two triangles and one quadrilateral
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
    nodes = [0, 1, 2, 3, 4, 5]
    xdofs = np.array([0, 1, 2, 3, 4, 5], dtype=int)
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

    ufl_e = basix.ufl.element("P", basix.CellType.prism, 2)
    form_compiler_options = {"scalar_type": np.float64}
    (ufcx_element, ufcx_dofmap), module, code = jit.ffcx_jit(
        mesh.comm, ufl_e, form_compiler_options=form_compiler_options
    )
    ffi = module.ffi
    cpp_dofmap = _cpp.fem.create_dofmaps(
        mesh.comm, [ffi.cast("uintptr_t", ffi.addressof(ufcx_dofmap))], mesh.topology
    )

    q = DofMap(cpp_dofmap[0])
    print(q.list)

    # el, dm = create_element_dofmap(mesh)
    # print()
    # for e,d in zip(el, dm):
    #     print(e.basix_element.cell_type.name)
    #     q = DofMap(d)
    #     print(q.index_map.size_local)
    #     print(q.list)
    #     print(q.dof_layout.entity_dofs(2, 0))
