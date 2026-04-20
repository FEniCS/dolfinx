from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import basix
import dolfinx.cpp as _cpp
import ufl
from dolfinx.cpp.fem import locate_dofs_geometrical
from dolfinx.cpp.mesh import GhostMode, create_mesh
from dolfinx.fem import (
    FiniteElement,
    FunctionSpace,
    assemble_matrix,
    assemble_vector,
    coordinate_element,
    create_dofmaps,
    dirichletbc,
    mixed_topology_form,
)
from dolfinx.io.utils import cell_perm_vtk
from dolfinx.mesh import CellType, Mesh, Topology, create_cell_partitioner

# -

if MPI.COMM_WORLD.size > 1:
    print("Not yet running in parallel")
    exit(0)


# ## Create a mixed-topology mesh

# +
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

part = create_cell_partitioner(GhostMode.none, 2)  # type: ignore
mesh = create_mesh(
    MPI.COMM_WORLD, cells_np, [hexahedron._cpp_object, prism._cpp_object], geomx, part, 2
)
import dolfinx

ud0 = ufl.Mesh(basix.ufl.element("Lagrange", "hexahedron", 1, shape=(3,)))
ud1 = ufl.Mesh(basix.ufl.element("Lagrange", "prism", 1, shape=(3,)))
ms = ufl.MeshSequence([ud0, ud1])

py_mesh = dolfinx.mesh.Mesh(mesh, ms)

elements = [
    basix.ufl.element("Lagrange", "hexahedron", 1, shape=(3,)),
    basix.ufl.element("Lagrange", "prism", 1, shape=(3,)),
]

W = dolfinx.fem.functionspace(py_mesh, elements)
u = dolfinx.fem.Function(W)
# -

# ## Create a mixed-topology dofmap and function space
# Create elements and dofmaps for each cell type

# +


# Select some BCs
def marker(x):
    """BC Selector."""
    return np.logical_or(np.isclose(x[2], 0.0), np.isclose(x[2], 1.0))


V_cpp = W.ufl_sub_space(0)._cpp_object
bcdofs = locate_dofs_geometrical(V_cpp, marker)
u_bc = dolfinx.fem.Function(W)
bc = dirichletbc(value=u_bc, dofs=bcdofs)

# -

# ## Creating and compiling a variational formulation
# We create the variational forms for each cell type.
# FIXME: This hack is required at the moment because UFL does not yet know
# about mixed topology meshes.

a = []
L = []

u_hex, u_prism = ufl.TrialFunctions(W)
v_hex, v_prism = ufl.TestFunctions(W)
prism_domain, hex_domain = py_mesh.ufl_domain().meshes
k = 12.0
dx_hex = ufl.Measure("dx", domain=hex_domain)
dx_prism = ufl.Measure("dx", domain=prism_domain)
a = ufl.inner(ufl.grad(u_hex), ufl.grad(v_hex)) * dx_hex - k**2 * ufl.inner(u_hex, v_hex) * dx_hex
a += (
    ufl.inner(ufl.grad(u_prism), ufl.grad(v_prism)) * dx_prism
    - k**2 * ufl.inner(u_prism, v_prism) * dx_prism
)

x_hex = ufl.SpatialCoordinate(hex_domain)
f_hex = ufl.as_vector([ufl.sin(ufl.pi * x_hex[0]) * ufl.sin(ufl.pi * x_hex[1]), 0.0, 0.0])
x_prism = ufl.SpatialCoordinate(prism_domain)
f_prism = ufl.as_vector([ufl.sin(ufl.pi * x_prism[0]) * ufl.sin(ufl.pi * x_prism[1]), 0.0, 0.0])
L = ufl.inner(f_hex, v_hex) * dx_hex
L += ufl.inner(f_prism, v_prism) * dx_prism


# Compile the form
# FIXME: For the time being, since UFL doesn't understand mixed topology
# meshes, we have to call {py:meth}`mixed_topology_form
# <dolfinx.fem.mixed_topology_form>` instead of form.

a_form = mixed_topology_form(a, dtype=np.float64)
L_form = mixed_topology_form(L, dtype=np.float64)

# ## Assembling and solving the linear system
# We use the native {py:class}`matrix<dolfinx.la.MatrixCSR>` and
# {py:class}`vector<dolfinx.la.Vector>` format in DOLFINx to assemble
# the left and right hand side of the linear system.

A = assemble_matrix(a_form, bcs=[bc])
b = assemble_vector(L_form)
bc.set(b.array)

# We use {py:func}`scipy.sparse.linalg.spsolve` to solve the
# resulting linear system

A_scipy = A.to_scipy()
b_scipy = b.array

x = spsolve(A_scipy, b_scipy)

print(f"Solution vector norm {np.linalg.norm(x)}")

# Mixed-topology I/O
# We manually build a ASCII XDMF file to store the mesh
# and solution
# NOTE: this should be replaced with VTKHDF

# +
xdmf = """<?xml version="1.0"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0" xmlns:xi="https://www.w3.org/2001/XInclude">
  <Domain>
    <Grid Name="mesh" GridType="Collection" CollectionType="spatial">

"""

perm = [cell_perm_vtk(CellType.hexahedron, 8), cell_perm_vtk(CellType.prism, 6)]
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
# -
