from mpi4py import MPI

import numpy as np
from scipy.sparse.linalg import spsolve

import basix
import dolfinx.cpp as _cpp
import ffcx
import ufl
from dolfinx import jit
from dolfinx.cpp.mesh import Mesh_float64, create_geometry, create_topology
from dolfinx.fem import DofMap, coordinate_element
from dolfinx.io.utils import cell_perm_vtk
from dolfinx.la import matrix_csr
from dolfinx.mesh import CellType

if MPI.COMM_WORLD.size > 1:
    raise RuntimeError("Serial only")


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


nx = 16
ny = 16
nz = 16
n_cells = nx * ny * nz
cells = [[], []]
orig_idx = [[], []]
idx = 0
for i in range(n_cells):
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
        cells[0] += [v0, v1, v2, v3, v4, v5, v6, v7]
        orig_idx[0] += [idx]
        idx += 1
    else:
        cells[1] += [v0, v1, v2, v4, v5, v6]
        orig_idx[1] += [idx]
        idx += 1
        cells[1] += [v1, v2, v3, v5, v6, v7]
        orig_idx[1] += [idx]
        idx += 1

n_points = (nx + 1) * (ny + 1) * (nz + 1)
sqxy = (nx + 1) * (ny + 1)
geom = []
for v in range(n_points):
    iz = v // sqxy
    p = v % sqxy
    iy = p // (nx + 1)
    ix = p % (nx + 1)
    geom += [[ix / nx, iy / ny, iz / nz]]
geomx = np.array(geom, dtype=np.float64)

ghost_owners = [[], []]
boundary_vertices = []

topology = create_topology(
    MPI.COMM_SELF,
    [CellType.hexahedron, CellType.prism],
    cells,
    orig_idx,
    ghost_owners,
    boundary_vertices,
)

entity_types = topology.entity_types
print(entity_types)

hexahedron = coordinate_element(CellType.hexahedron, 1)
prism = coordinate_element(CellType.prism, 1)
print(geomx.shape)
nodes = np.arange(geomx.shape[0], dtype=np.int64)
xdofs = np.array(cells[0] + cells[1], dtype=np.int64)

geom = create_geometry(
    topology, [hexahedron._cpp_object, prism._cpp_object], nodes, xdofs, geomx.flatten(), 3
)

mesh = Mesh_float64(MPI.COMM_WORLD, topology, geom)

# Order 1 dofmaps
elements, dofmaps = create_element_dofmap(
    mesh, [basix.CellType.hexahedron, basix.CellType.prism], 1
)
q = [DofMap(dofmaps[0]), DofMap(dofmaps[1])]

# Both dofmaps have the same IndexMap, but different cell_dofs
# Create SparsityPattern
sp = _cpp.la.SparsityPattern(MPI.COMM_WORLD, [q[0].index_map, q[0].index_map], [1, 1])
for ct in range(2):
    num_cells_type = mesh.topology.index_maps(3)[ct].size_local
    print(ct, num_cells_type)
    for j in range(num_cells_type):
        cell_dofs_j = q[ct].cell_dofs(j)
        sp.insert(cell_dofs_j, cell_dofs_j)
sp.finalize()


def get_compiled_form(cell_name):
    element = basix.ufl.element("Lagrange", cell_name, 1)
    domain = ufl.Mesh(basix.ufl.element("Lagrange", cell_name, 1, shape=(3,)))
    space = ufl.FunctionSpace(domain, element)
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    k = 0.1
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) + k * u * v) * ufl.dx
    forms = [a]
    return ffcx.codegeneration.jit.compile_forms(forms, options={"scalar_type": np.float64})


cf_hex, module, _ = get_compiled_form("hexahedron")
cf_prism, module, _ = get_compiled_form("prism")
ffi = module.ffi
kernels = [
    getattr(cf_cell[0].form_integrals[0], "tabulate_tensor_float64")
    for cf_cell in [cf_hex, cf_prism]
]


# Assembler
A = matrix_csr(sp)
print(A)

# For each cell type
for ct in range(2):
    num_cells_type = mesh.topology.index_maps(3)[ct].size_local
    geom_dm = mesh.geometry.dofmaps(ct)
    kernel = kernels[ct]
    for j in range(num_cells_type):
        cell_dofs_j = q[ct].cell_dofs(j)
        A_local = np.zeros((len(cell_dofs_j) ** 2), dtype=np.float64)
        cell_geom = mesh.geometry.x[geom_dm[j]]
        kernel(
            ffi.cast("double *", A_local.ctypes.data),
            ffi.NULL,
            ffi.NULL,
            ffi.cast("double *", cell_geom.ctypes.data),
            ffi.NULL,
            ffi.NULL,
        )
        A.add(A_local, cell_dofs_j, cell_dofs_j, 1)


A_scipy = A.to_scipy()
b_scipy = np.ones(A_scipy.shape[1])

x = spsolve(A_scipy, b_scipy)
print(x)

# I/O
# Save to XDMF
xdmf = """<?xml version="1.0"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0" xmlns:xi="https://www.w3.org/2001/XInclude">
  <Domain>
    <Grid Name="mesh" GridType="Uniform">
"""

vtk_topology = []
codes = [9, 8]
perm = [cell_perm_vtk(CellType.hexahedron, 8), [0, 1, 2, 3, 4, 5]]

for i in range(2):
    geom_dm = mesh.geometry.dofmaps(i)
    for c in geom_dm:
        vtk_topology += [codes[i]]
        vtk_topology += list(c[perm[i]])

xdmf += f"""
      <Topology TopologyType="Mixed">
        <DataItem Dimensions="{len(vtk_topology)}" Precision="4" NumberType="Int" Format="XML">
          {" ".join(str(val) for val in vtk_topology)}
        </DataItem>
      </Topology>"""

xdmf += f"""
      <Geometry GeometryType="XYZ" NumberType="float" Rank="2" Precision="8">
        <DataItem Dimensions="{mesh.geometry.x.shape[0]} 3" Format="XML">
          {" ".join(str(val) for val in mesh.geometry.x.flatten())}
        </DataItem>
      </Geometry>"""

xdmf += f"""
      <Attribute Name="u" Center="Node" NumberType="float" Precision="8">
        <DataItem Dimensions="{len(x)}" Format="XML">
          {" ".join(str(val) for val in x)}
        </DataItem>
      </Attribute>"""

xdmf += """
    </Grid>
  </Domain>
</Xdmf>
"""

fd = open("mixed-mesh.xdmf", "w")
fd.write(xdmf)
fd.close()
