from mpi4py import MPI

import numpy as np
from scipy.sparse.linalg import spsolve

import basix
import dolfinx.cpp as _cpp
import ufl
from dolfinx.cpp.mesh import GhostMode, create_cell_partitioner, create_mesh
from dolfinx.fem import (
    FunctionSpace,
    assemble_matrix,
    coordinate_element,
    form,
)
from dolfinx.io.utils import cell_perm_vtk
from dolfinx.mesh import CellType, Mesh

if MPI.COMM_WORLD.size > 1:
    print("Not yet running in parallel")
    exit(0)


# Create a mesh
nx = 16
ny = 16
nz = 16
n_cells = nx * ny * nz

cells: list = [[], []]
orig_idx: list = [[], []]
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
    for v in range(n_points):
        iz = v // sqxy
        p = v % sqxy
        iy = p // (nx + 1)
        ix = p % (nx + 1)
        geom += [[ix / nx, iy / ny, iz / nz]]

cells_np = [np.array(c) for c in cells]
geomx = np.array(geom, dtype=np.float64)
hexahedron = coordinate_element(CellType.hexahedron, 1)
prism = coordinate_element(CellType.prism, 1)

part = create_cell_partitioner(GhostMode.none)
mesh = create_mesh(
    MPI.COMM_WORLD, cells_np, [
        hexahedron._cpp_object, prism._cpp_object], geomx, part
)

# Create order 1 dofmaps on mesh
elements = [
    basix.create_element(basix.ElementFamily.P, basix.CellType.hexahedron, 1),
    basix.create_element(basix.ElementFamily.P, basix.CellType.prism, 1),
]

elements_cpp = [_cpp.fem.FiniteElement_float64(
    e._e, None, True) for e in elements]
dofmaps = _cpp.fem.create_dofmaps(mesh.comm, mesh.topology, elements_cpp)
V_cpp = _cpp.fem.FunctionSpace_float64(mesh, elements_cpp, dofmaps)
# Both dofmaps have the same IndexMap, but different cell_dofs

# Create forms for each cell type
a = []
for i, cell_name in enumerate(["hexahedron", "prism"]):
    print(f"Creating form for {cell_name}")
    element = basix.ufl.wrap_element(elements[i])
    domain = ufl.Mesh(basix.ufl.element("Lagrange", cell_name, 1, shape=(3,)))
    V = FunctionSpace(Mesh(mesh, domain), element, V_cpp)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    k = 12.0
    a += [(ufl.inner(ufl.grad(u), ufl.grad(v)) - k**2 * u * v) * ufl.dx]

a_form = form(a)

A = assemble_matrix(a_form)

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

perm = [cell_perm_vtk(CellType.hexahedron, 8),
        cell_perm_vtk(CellType.prism, 6)]
topologies = ["Hexahedron", "Wedge"]

for j in range(2):
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
