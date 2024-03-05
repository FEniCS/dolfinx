from mpi4py import MPI

import numpy as np
from scipy.sparse.linalg import spsolve

import basix
import dolfinx.cpp as _cpp
import ffcx
import ufl
from dolfinx import jit
from dolfinx.cpp.la import SparsityPattern
from dolfinx.cpp.mesh import Mesh_float64, create_geometry, create_topology
from dolfinx.fem import coordinate_element
from dolfinx.fem.dofmap import DofMap
from dolfinx.io.utils import cell_perm_vtk
from dolfinx.la import matrix_csr
from dolfinx.log import LogLevel, set_log_level
from dolfinx.mesh import CellType


def create_kernel(cell, degree):
    element = basix.ufl.element("Lagrange", cell, degree)
    gdim = element.cell.topological_dimension()
    domain = ufl.Mesh(basix.ufl.element("Lagrange", cell, 1, shape=(gdim,)))
    space = ufl.FunctionSpace(domain, element)
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)

    a = -0.2 * ufl.inner(u, v) * ufl.dx
    a += ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    f, mod, code = ffcx.codegeneration.jit.compile_forms(
        [a], options={"scalar_type": "float64", "cache_dir": "."}
    )
    form = f[0]
    integral = form.form_integrals[form.form_integral_offsets[mod.lib.cell]]
    kernel = integral.tabulate_tensor_float64
    return kernel, mod.ffi


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


def test_assemble():
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

    im = dm[0].index_map
    sp = SparsityPattern(mesh.comm, [im, im], [1, 1])
    for ei, di in zip(el, dm):
        dofs = DofMap(di).list
        for j in range(dofs.shape[0]):
            sp.insert(dofs[j, :], dofs[j, :])
    sp.finalize()

    A = matrix_csr(sp)

    kernels = [create_kernel("triangle", 2), create_kernel("quadrilateral", 2)]

    for i, (ei, di) in enumerate(zip(el, dm)):
        geom = mesh.geometry.dofmaps(i)
        dofs = DofMap(di).list
        for j in range(dofs.shape[0]):
            # Cell geometry
            coords = mesh.geometry.x[geom[j], :].flatten()
            vals = np.zeros(dofs.shape[1] ** 2, dtype=np.float64)

            # kernel(vals, coords)
            k, ffi = kernels[i]
            k(
                ffi.cast("double *", vals.ctypes.data),
                ffi.NULL,
                ffi.NULL,
                ffi.cast("double *", coords.ctypes.data),
                ffi.NULL,
                ffi.NULL,
            )
            A.add(vals, dofs[j, :], dofs[j, :])

    A.scatter_reverse()

    print(sum(A.data))

    As = A.to_scipy()
    As[0, 1:] = 0.0
    As[0, 0] = 1.0
    b = np.ones(As.shape[1], dtype=np.float64)
    x = spsolve(As, b)
    print(x)


def test_3d_mixed_assemble():
    nx = 16
    ny = 16
    nz = 16

    geom = []
    sqxy = (nx + 1) * (ny + 1)
    for v in range((nx + 1) * (ny + 1) * (nz + 1)):
        iz = v // sqxy
        p = v % sqxy
        iy = p // (nx + 1)
        ix = p % (nx + 1)
        geom += [ix, iy, iz]
    nodes = np.arange((nx + 1) * (ny + 1) * (nz + 1), dtype=np.int64)

    hex_cells = []
    prism_cells = []
    for i in range(nx * ny * nz):
        iz = i // (nx * ny)
        j = i % (nx * ny)
        iy = j // nx
        ix = j % nx

        v0 = (iz * (ny + 1) + iy) * (nx + 1) + ix
        v1 = v0 + 1
        v2 = v0 + (nx + 1)
        v3 = v1 + (nx + 1)
        v4 = v0 + (nx + 1) * (ny + 1)
        v5 = v1 + (nx + 1) * (ny + 1)
        v6 = v2 + (nx + 1) * (ny + 1)
        v7 = v3 + (nx + 1) * (ny + 1)

        if ix < nx / 2:
            hex_cells += [v0, v1, v2, v3, v4, v5, v6, v7]
        else:
            prism_cells += [v1, v3, v0, v5, v7, v4]
            prism_cells += [v3, v0, v2, v7, v4, v6]
            # tet_cells += [v0, v1, v3, v7,
            #               v0, v1, v7, v5,
            #               v0, v5, v7, v4,
            #               v0, v3, v2, v7,
            #               v0, v6, v4, v7,
            #               v0, v2, v6, v7]

    cells = [hex_cells, prism_cells]
    hex_index = [i for i in range(len(hex_cells))]
    prism_index = [i + hex_index[-1] for i in range(len(prism_cells))]

    # cells with global indexing
    orig_index = [hex_index, prism_index]
    # No ghosting
    ghost_owners = [[], []]
    # All vertices are potentially on boundary
    boundary_vertices = [i for i in range(nx * ny * nz)]

    topology = create_topology(
        MPI.COMM_WORLD,
        [CellType.hexahedron, CellType.prism],
        cells,
        orig_index,
        ghost_owners,
        boundary_vertices,
    )

    hex = coordinate_element(CellType.hexahedron, 1)
    prism = coordinate_element(CellType.prism, 1)
    xdofs = np.array(hex_cells + prism_cells, dtype=np.int64)
    x = np.array(geom, dtype=np.float64)
    geom = create_geometry(topology, [hex._cpp_object, prism._cpp_object], nodes, xdofs, x, 3)
    mesh = Mesh_float64(MPI.COMM_WORLD, topology, geom)

    print(mesh)

    # First order dofmap on mixed mesh
    el, dm = create_element_dofmap(mesh, [basix.CellType.hexahedron, basix.CellType.prism], 1)

    im = dm[0].index_map
    sp = SparsityPattern(mesh.comm, [im, im], [1, 1])
    for ei, di in zip(el, dm):
        dofs = DofMap(di).list
        for j in range(dofs.shape[0]):
            sp.insert(dofs[j, :], dofs[j, :])
    sp.finalize()

    A = matrix_csr(sp)

    kernels = [create_kernel("hexahedron", 1), create_kernel("prism", 1)]
    dofmaps = [DofMap(di) for di in dm]

    for i, (ei, di) in enumerate(zip(el, dofmaps)):
        geom = mesh.geometry.dofmaps(i)
        dofs = di.list
        for j in range(dofs.shape[0]):
            # Cell geometry
            coords = mesh.geometry.x[geom[j], :].flatten()
            vals = np.zeros(dofs.shape[1] ** 2, dtype=np.float64)
            # kernel(vals, coords)
            k, ffi = kernels[i]
            k(
                ffi.cast("double *", vals.ctypes.data),
                ffi.NULL,
                ffi.NULL,
                ffi.cast("double *", coords.ctypes.data),
                ffi.NULL,
                ffi.NULL,
            )
            A.add(vals, dofs[j, :], dofs[j, :])

    A.scatter_reverse()

    print(sum(A.data))

    As = A.to_scipy()
    b = np.ones(As.shape[1], dtype=np.float64)
    x = spsolve(As, b)
    print(x.max(), x.min(), np.linalg.norm(x))

    # Output as an XDMF mesh

    topology_list = []

    cell_to_vtk = {basix._basixcpp.CellType.prism: 8, basix._basixcpp.CellType.hexahedron: 9}

    attr_data = np.zeros(len(x))

    for i, (ei, di) in enumerate(zip(el, dm)):
        cell_type = cell_to_vtk.get(ei.basix_element.cell_type)
        geom = mesh.geometry.dofmaps(i)
        dofs = DofMap(di).list
        reord = np.arange(geom.shape[1], dtype=int)
        if i == 0:
            reord = cell_perm_vtk(CellType.hexahedron, 8)
        elif i == 1:
            reord = cell_perm_vtk(CellType.prism, 6)
        for j in range(dofs.shape[0]):
            for gx, dx in zip(geom[j], dofs[j]):
                attr_data[gx] = x[dx]
            # Cell geometry
            topology_list += [cell_type, *geom[j, reord]]

    print(topology_list)

    point_data = " ".join(str(xi) for xi in mesh.geometry.x.flatten())
    num_points = mesh.geometry.x.shape[0]

    xdmf = f"""
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">
<Domain>
  <Grid Name="MixedMesh" GridType="Uniform">
    <Topology TopologyType="Mixed">
      <DataItem Dimensions="{len(topology_list)}" NumberType="Int" Precision="4" Format="XML">
{" ".join([str(x) for x in topology_list])}
      </DataItem>
    </Topology>
    <Geometry GeometryType="XYZ">
      <DataItem Rank="2" Dimensions="{num_points} 3" NumberType="Float" Precision="8" Format="XML">
{point_data}
      </DataItem>
    </Geometry>
    <Attribute Name="u" Center="Node">
    <DataItem Format="XML" Dimensions="{len(attr_data)}">
      {" ".join([str(val) for val in attr_data])}
      </DataItem>
    </Attribute>
  </Grid>
</Domain>
</Xdmf>
"""

    f = open("a.xdmf", "w")
    f.write(xdmf)
    f.close()
