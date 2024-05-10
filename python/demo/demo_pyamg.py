# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Poisson equation with pyamg solver
#
# This demo is implemented in {download}`demo_pyamg.py`. It
# illustrates how to:
#
# - Solve the Poisson equation with pyamg
#

from mpi4py import MPI

# +
import numpy as np

try:
    import pyamg
except ImportError:
    print("This demo requires pyamg.")
    exit(0)

from scipy.sparse import csr_matrix

import ufl
from dolfinx import fem, io
from dolfinx.fem import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
    set_bc,
)
from dolfinx.mesh import CellType, create_box, locate_entities_boundary
from ufl import ds, dx, grad, inner

dtype = np.float64
if MPI.COMM_WORLD.size > 1:
    raise RuntimeError("Only works in serial.")

mesh = create_box(
    comm=MPI.COMM_WORLD,
    points=[(0.0, 0.0, 0.0), (3.0, 2.0, 1.0)],
    n=[96, 64, 32],
    cell_type=CellType.tetrahedron,
    dtype=dtype,
)
V = functionspace(mesh, ("Lagrange", 1))

facets = locate_entities_boundary(
    mesh,
    dim=(mesh.topology.dim - 1),
    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 3.0),
)

tdim = mesh.topology.dim
dofs = locate_dofs_topological(V=V, entity_dim=tdim - 1, entities=facets)

bc = dirichletbc(value=dtype(0), dofs=dofs, V=V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(mesh)
f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
g = ufl.sin(5 * x[0])
a = form(inner(grad(u), grad(v)) * dx, dtype=dtype)
L = form(inner(f, v) * dx + inner(g, v) * ds, dtype=dtype)

A0 = assemble_matrix(a, [bc])
b = assemble_vector(L)
apply_lifting(b.array, [a], bcs=[[bc]])
set_bc(b.array, [bc])

A = csr_matrix((A0.data, A0.indices, A0.indptr))
uh = fem.Function(V, dtype=dtype)
ml = pyamg.ruge_stuben_solver(A)
print(ml)

res: list[float] = []
uh.x.array[:] = ml.solve(b.array, tol=1e-10, residuals=res, accel="cg")
for i, q in enumerate(res):
    print(f"Iteration {i}, residual= {q}")

with io.XDMFFile(mesh.comm, "out_pyamg/poisson.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(uh)
