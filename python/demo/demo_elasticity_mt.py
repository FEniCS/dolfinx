from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import basix
import dolfinx.cpp as _cpp
import ufl
from dolfinx.cpp.fem import locate_dofs_geometrical
from dolfinx.cpp.mesh import GhostMode, create_mesh
from dolfinx.fem import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    coordinate_element,
    dirichletbc,
)
from dolfinx.io.utils import cell_perm_vtk
from dolfinx.mesh import CellType, create_cell_partitioner
from scipy.sparse.linalg import spsolve
# -

if MPI.COMM_WORLD.size > 1:
    print("Not yet running in parallel")
    exit(0)


# ## Create a mixed-topology mesh

# +
nx = 10
ny = 11
nz = 12
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

ud0 = ufl.Mesh(
    basix.ufl.element(
        "Lagrange",
        "hexahedron",
        1,
        shape=(3,),
    )
)
ud1 = ufl.Mesh(
    basix.ufl.element(
        "Lagrange",
        "prism",
        1,
        shape=(3,),
    )
)
ud0._ufl_cargo = mesh
ud1._ufl_cargo = mesh
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
    return np.isclose(x[0], 0.0)


V_cpp = W.ufl_sub_space(0)._cpp_object
bcdofs = locate_dofs_geometrical(V_cpp, marker)
u_bc = dolfinx.fem.Function(W)
bc = dirichletbc(value=u_bc, dofs=bcdofs)


def marker2(x):
    """BC Selector."""
    return np.isclose(x[0], 1.0)


bcdofs2 = locate_dofs_geometrical(V_cpp, marker2)
u_bc2 = dolfinx.fem.Function(W)
u_bc2.x.array[:] = 0.00
bc2 = dirichletbc(value=u_bc2, dofs=bcdofs2)

# -

# ## Creating and compiling a variational formulation
# We create the variational forms for each cell type.
# FIXME: This hack is required at the moment because UFL does not yet know
# about mixed topology meshes.

a = []
L = []

u_hex, u_prism = ufl.TrialFunctions(W)
v_hex, v_prism = ufl.TestFunctions(W)
hex_domain, prism_domain = py_mesh.ufl_domain().meshes

dx_hex = ufl.Measure("dx", domain=hex_domain)
dx_prism = ufl.Measure("dx", domain=prism_domain)

lambda_ = 1.0e8
mu = 0.4


def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)


def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)


scale = 0.01
a = ufl.inner(sigma(u_hex), epsilon(v_hex)) * dx_hex
a += ufl.inner(sigma(u_prism), epsilon(v_prism)) * dx_prism
L = ufl.inner(ufl.as_vector((0.0, 0.0, scale * -9.81)), v_hex) * dx_hex
L += ufl.inner(ufl.as_vector((0.0, 0.0, scale * -9.81)), v_prism) * dx_prism

a_form = dolfinx.fem.form(a, dtype=np.float64)
L_form = dolfinx.fem.form(L, dtype=np.float64)

# ## Assembling and solving the linear system
# We use the native {py:class}`matrix<dolfinx.la.MatrixCSR>` and
# {py:class}`vector<dolfinx.la.Vector>` format in DOLFINx to assemble
# the left and right hand side of the linear system.

bcs = [bc, bc2]
A = assemble_matrix(a_form, bcs=bcs)
b = assemble_vector(L_form)
apply_lifting(b.array, [a_form], bcs=[bcs])
for bc in bcs:
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
    # num_vertices = dolfinx.cpp.mesh.cell_num_vertices(mesh.topology.cell_types[j])
    # dl = W.ufl_sub_space(j).dofmaps(j).dof_layout
    # vertex_pos = np.array([dl.entity_dofs(0, i) for i in range(num_vertices)]).flatten()
    # dofs = W.ufl_sub_space(j).dofmaps(j).list[:, vertex_pos]

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
          <DataItem Dimensions="{len(x) / 3} 3" Format="XML">
            {" ".join(str(val) for val in x)}
          </DataItem>
       </Attribute>
      </Grid>"""

xdmf += """
    </Grid>
  </Domain>
</Xdmf>
"""

fd = open("mixed-mesh2.xdmf", "w")
fd.write(xdmf)
fd.close()
# -
