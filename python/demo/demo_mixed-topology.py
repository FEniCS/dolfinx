# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Poisson equation
#
# This demo illustrates how to solve a simple Helmholtz problem on a
# mixed-topology mesh.
#
# NOTE: Mixed-topology meshes are a work in progress and are not yet fully
# supported in DOLFINx.

from mpi4py import MPI
from petsc4py import PETSc

import h5py
import numpy as np

import basix
import dolfinx.cpp as _cpp
import ufl
from dolfinx.cpp.mesh import GhostMode, create_cell_partitioner, create_mesh
from dolfinx.fem import (
    FunctionSpace,
    coordinate_element,
    mixed_topology_form,
)
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.io.vtkhdf import write_mesh
from dolfinx.la import create_petsc_vector
from dolfinx.mesh import CellType, Mesh

# Create a mixed-topology mesh
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
    geomx = np.array(geom, dtype=np.float64)
else:
    geomx = np.zeros((0, 3), dtype=np.float64)

cells_np = [np.array(c) for c in cells]
hexahedron = coordinate_element(CellType.hexahedron, 1)
prism = coordinate_element(CellType.prism, 1)

part = create_cell_partitioner(GhostMode.none)
mesh = create_mesh(
    MPI.COMM_WORLD, cells_np, [hexahedron._cpp_object, prism._cpp_object], geomx, part
)

# Create elements and dofmaps for each cell type
elements = [
    basix.create_element(basix.ElementFamily.P, basix.CellType.hexahedron, 1),
    basix.create_element(basix.ElementFamily.P, basix.CellType.prism, 1),
]
elements_cpp = [_cpp.fem.FiniteElement_float64(e._e, None, True) for e in elements]
# NOTE: Both dofmaps have the same IndexMap, but different cell_dofs
dofmaps = _cpp.fem.create_dofmaps(mesh.comm, mesh.topology, elements_cpp)

# Create C++ function space
V_cpp = _cpp.fem.FunctionSpace_float64(mesh, elements_cpp, dofmaps)

# Create forms for each cell type.
# FIXME This hack is required at the moment because UFL does not yet know about
# mixed topology meshes.
a = []
L = []
for i, cell_name in enumerate(["hexahedron", "prism"]):
    print(f"Creating form for {cell_name}")
    element = basix.ufl.wrap_element(elements[i])
    domain = ufl.Mesh(basix.ufl.element("Lagrange", cell_name, 1, shape=(3,)))
    V = FunctionSpace(Mesh(mesh, domain), element, V_cpp)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    k = 12.0
    a += [(ufl.inner(ufl.grad(u), ufl.grad(v)) - k**2 * u * v) * ufl.dx]
    L += [v * ufl.dx]

# Compile the form
# FIXME: For the time being, since UFL doesn't understand mixed topology meshes,
# we have to call mixed_topology_form instead of form.
a_form = mixed_topology_form(a, dtype=np.float64)
L_form = mixed_topology_form(L, dtype=np.float64)

# Assemble the matrix
A = assemble_matrix(a_form)
# b = assemble_vector(L_form)
A.assemble()
im = V_cpp.dofmaps(0).index_map
u = create_petsc_vector(im, 1)
b = create_petsc_vector(im, 1)

b.array[:] = 1.0

ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A)
ksp.solve(b, u)

print(u.norm())

write_mesh("aa.vtkhdf", Mesh(mesh, None))
b = h5py.File("aa.vtkhdf", "a", driver="mpio", comm=mesh.comm)
vtkhdf = b["VTKHDF"]
pd = vtkhdf.create_group("PointData")
uvtk = pd.create_dataset("u", (im.size_global,))
r = im.local_range
uvtk[r[0] : r[1]] = u.array[: im.size_local]
b.close()
