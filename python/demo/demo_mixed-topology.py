from pathlib import Path

from mpi4py import MPI

import numpy as np
from read_write_vtkhdf import write
from scipy.sparse.linalg import spsolve

import basix
import dolfinx.cpp as _cpp
import ufl
from dolfinx.cpp.mesh import GhostMode, create_cell_partitioner, create_mesh
from dolfinx.fem import FunctionSpace, coordinate_element, form
from dolfinx.io.utils import cell_perm_vtk
from dolfinx.la import matrix_csr
from dolfinx.mesh import CellType, Mesh

if MPI.COMM_WORLD.size > 4:
    print("Not yet running in parallel")
    exit(0)


# Create a mesh
nx = 25
ny = 23
nz = 21
n_cells = nx * ny * nz

cells: list = [[], [], [], []]
orig_idx: list = [[], [], [], []]
geom = []

if MPI.COMM_WORLD.rank == 0:
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

        if iz < nz / 2:
            if (ix < nx / 2 and iy < ny / 2) or (ix >= nx / 2 and iy >= ny / 2):
                cells[0] += [v0, v1, v2, v3, v4, v5, v6, v7]
                orig_idx[0] += [idx]
                idx += 1
            else:
                cells[1] += [v0, v1, v3, v4, v5, v7]
                orig_idx[1] += [idx]
                idx += 1
                cells[1] += [v0, v2, v3, v4, v6, v7]
                orig_idx[1] += [idx]
                idx += 1
        else:
            if (iy < ny / 2 and ix >= nx / 2) or (iy >= ny / 2 and ix < nx / 2):
                cells[2] += [v0, v1, v3, v7]
                orig_idx[2] += [idx]
                idx += 1
                cells[2] += [v0, v1, v7, v5]
                orig_idx[2] += [idx]
                idx += 1
                cells[2] += [v0, v5, v7, v4]
                orig_idx[2] += [idx]
                idx += 1
                cells[2] += [v0, v3, v2, v7]
                orig_idx[2] += [idx]
                idx += 1
                cells[2] += [v0, v6, v4, v7]
                orig_idx[2] += [idx]
                idx += 1
                cells[2] += [v0, v2, v6, v7]
                orig_idx[2] += [idx]
                idx += 1
            else:
                cells[3] += [v0, v1, v2, v3, v7]
                orig_idx[3] += [idx]
                idx += 1
                cells[3] += [v0, v1, v4, v5, v7]
                orig_idx[3] += [idx]
                idx += 1
                cells[3] += [v0, v2, v4, v6, v7]
                orig_idx[3] += [idx]
                idx += 1

    n_points = (nx + 1) * (ny + 1) * (nz + 1)
    sqxy = (nx + 1) * (ny + 1)
    for v in range(n_points):
        iz = v // sqxy
        p = v % sqxy
        iy = p // (nx + 1)
        ix = p % (nx + 1)
        geom += [[ix / nx, iy / ny, iz / nz]]

cells_np = [np.array(c) for c in cells]
geomx = np.array(geom, dtype=np.float64)
if len(geom) == 0:
    geomx = np.empty((0, 3), dtype=np.float64)
else:
    geomx = np.array(geom, dtype=np.float64)

cell_types = [CellType.hexahedron, CellType.prism, CellType.tetrahedron, CellType.pyramid]
coordinate_elements = [coordinate_element(cell, 1) for cell in cell_types]
part = create_cell_partitioner(GhostMode.none)
mesh = create_mesh(
    MPI.COMM_WORLD, cells_np, [e._cpp_object for e in coordinate_elements], geomx, part
)

# Create order 1 dofmaps on mesh
elements = [basix.ufl.element("Lagrange", cell.name, 1) for cell in cell_types]

cpp_elements = [_cpp.fem.FiniteElement_float64(e.basix_element._e, None, True) for e in elements]
dofmaps = _cpp.fem.create_dofmaps(mesh.comm, mesh.topology, cpp_elements)

# Both dofmaps have the same IndexMap, but different cell_dofs
# Create SparsityPattern
sp = _cpp.la.SparsityPattern(MPI.COMM_WORLD, [dofmaps[0].index_map, dofmaps[0].index_map], [1, 1])
for ct, dm in enumerate(dofmaps):
    num_cells_type = mesh.topology.index_maps(3)[ct].size_local
    print(f"For cell type {ct}, create sparsity with {num_cells_type} cells.")
    for j in range(num_cells_type):
        cell_dofs_j = dm.cell_dofs(j)
        sp.insert(cell_dofs_j, cell_dofs_j)
sp.finalize()

# Compile forms for each cell type
aforms = []
for element, cpp_element, dofmap in zip(elements, cpp_elements, dofmaps):
    print(f"Compiling form for {element.cell_type.name}")
    domain = ufl.Mesh(basix.ufl.element("Lagrange", element.cell_type, 1, shape=(3,)))
    cppV = _cpp.fem.FunctionSpace_float64(mesh, cpp_element, dofmap)
    V = FunctionSpace(Mesh(mesh, domain), element, cppV)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    k = 12.0
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - k**2 * u * v) * ufl.dx
    aforms += [form(a)]

ffi = aforms[0].module.ffi

# Assembler
A = matrix_csr(sp)
print(f"Assembling into matrix of size {len(A.data)} non-zeros")

# Assemble for each cell type (ct)
for ct, aform in enumerate(aforms):
    num_cells_type = mesh.topology.index_maps(3)[ct].size_local
    geom_dm = mesh.geometry.dofmaps(ct)
    kernel = getattr(aforms[ct].ufcx_form.form_integrals[0], "tabulate_tensor_float64")
    dm = aform.function_spaces[0].dofmap
    for j in range(num_cells_type):
        cell_dofs_j = dm.cell_dofs(j)
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

write(mesh, Path("mixed_mesh.vtkhdf"))

exit(0)


# Quick solve
A_scipy = A.to_scipy()
b_scipy = np.ones(A_scipy.shape[1])

x = spsolve(A_scipy, b_scipy)
print(f"Solution vector norm {np.linalg.norm(x)}")

# I/O
# Save to XDMF
xdmf = """<?xml version="1.0"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0" xmlns:xi="https://www.w3.org/2001/XInclude">
  <Domain>
    <Grid Name="mesh" GridType="Collection" CollectionType="spatial">

"""

perm = [
    cell_perm_vtk(CellType.hexahedron, 8),
    cell_perm_vtk(CellType.prism, 6),
    cell_perm_vtk(CellType.tetrahedron, 4),
    cell_perm_vtk(CellType.pyramid, 5),
]
topologies = ["Hexahedron", "Wedge", "Tetrahedron", "Pyramid"]

for j in range(len(topologies)):
    vtk_topology = []
    geom_dm = mesh.geometry.dofmaps(j)
    for c in geom_dm:
        vtk_topology += list(c[perm[j]])
    topology_type = topologies[j]

    xdmf += f"""
      <Grid Name="{topology_type}" GridType="Uniform">
        <Topology TopologyType="{topology_type}">
          <DataItem Dimensions="{geom_dm.shape[0]} {geom_dm.shape[1]}"
           Precision="4" NumberType="Int" Format="XML">
          {" ".join(str(val) for val in vtk_topology)}
          </DataItem>
        </Topology>
        <Geometry GeometryType="XYZ" NumberType="float" Rank="2" Precision="8">
          <DataItem Dimensions="{mesh.geometry.x.shape[0]} 3" Format="XML">
            {" ".join(str(val) for val in mesh.geometry.x.flatten())}
          </DataItem>
        </Geometry>
        <Attribute Name="u" Center="Node" NumberType="float" Precision="8">
          <DataItem Dimensions="{len(x)}" Format="XML">
            {" ".join(str(val) for val in x)}
          </DataItem>
       </Attribute>
      </Grid>"""

xdmf += """
    </Grid>
  </Domain>
</Xdmf>
"""

fd = open("mixed-mesh.xdmf", "w")
fd.write(xdmf)
fd.close()
