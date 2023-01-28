# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Solving PDEs with different scalar (float) types
#
# This demo  ({download}`demo_types.py`) shows:
#
# - How to solve problems using different scalar types, e.g. single or
#   double precision, or complex numbers
# - Interfacing with [SciPy](https://scipy.org/) sparse linear algebra
#   functionality

# +
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import ufl
from dolfinx import fem, la, mesh, plot

from mpi4py import MPI
# -

# SciPy solvers do not support MPI, so all computations will be
# performed on a single MPI rank

# +
comm = MPI.COMM_SELF
# -

# Create a mesh and function space.

msh = mesh.create_rectangle(comm=comm, points=((0.0, 0.0), (2.0, 1.0)), n=(32, 16),
                            cell_type=mesh.CellType.triangle)
V = fem.FunctionSpace(msh, ("Lagrange", 1))

# Define a variational problem.

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
fr = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
fc = ufl.sin(2 * np.pi * x[0]) + 10 * ufl.sin(4 * np.pi * x[1]) * 1j
gr = ufl.sin(5 * x[0])
gc = ufl.sin(5 * x[0]) * 1j
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(fr + fc, v) * ufl.dx + ufl.inner(gr + gc, v) * ufl.ds

# In preparation for constructing Dirichlet boundary conditions, locate
# facets on the constrained boundary and the corresponding
# degrees-of-freedom.

facets = mesh.locate_entities_boundary(msh, dim=1,
                                       marker=lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                      np.isclose(x[0], 2.0)))
dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)


# The below function computes the solution of the finite problem using a
# specified scalar type.

def solve_scalar(dtype=np.float32):
    """Solve the variational problem"""

    # Process forms. This will compile the forms for the requested type.
    a0 = fem.form(a, dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        L0 = fem.form(L, dtype=dtype)
    else:
        L0 = fem.form(ufl.replace(L, {fc: 0, gc: 0}), dtype=dtype)

    # Create a Dirichlet boundary condition
    bc = fem.dirichletbc(value=dtype(0), dofs=dofs, V=V)

    # Assemble forms
    A = fem.assemble_matrix(a0, [bc])
    A.finalize()
    b = fem.assemble_vector(L0)
    fem.apply_lifting(b.array, [a0], bcs=[[bc]])
    b.scatter_reverse(la.ScatterMode.add)
    fem.set_bc(b.array, [bc])

    # Create a Scipy sparse matrix that shares data with A
    As = scipy.sparse.csr_matrix((A.data, A.indices, A.indptr))

    # Solve the variational problem and return the solution
    uh = fem.Function(V, dtype=dtype)
    uh.x.array[:] = scipy.sparse.linalg.spsolve(As, b.array)
    return uh

# This function visualises the solution.


def display_scalar(u, filter=np.real):
    """Plot the solution using pyvista"""
    try:
        import pyvista
        cells, types, x = plot.create_vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(cells, types, x)
        grid.point_data["u"] = filter(u.x.array)
        grid.set_active_scalars("u")

        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_edges=True)
        plotter.add_mesh(grid.warp_by_scalar())
        plotter.add_title("real" if filter is np.real else "imag")
        if pyvista.OFF_SCREEN:
            pyvista.start_xvfb(wait=0.1)
            plotter.screenshot(f"u_{'real' if filter is np.real else 'imag'}.png")
        else:
            plotter.show()
    except ModuleNotFoundError:
        print("'pyvista' is required to visualise the solution")


# Solve the variational problem using different scalar types

uh = solve_scalar(dtype=np.float32)
uh = solve_scalar(dtype=np.float64)
uh = solve_scalar(dtype=np.complex64)
uh = solve_scalar(dtype=np.complex128)

# Display the last computed solution

display_scalar(uh, np.real)
display_scalar(uh, np.imag)


E = 1.0e9
ν = 0.3
μ = E / (2.0 * (1.0 + ν))
λ = E * ν / ((1.0 + ν) * (1.0 - 2.0 * ν))


def σ(v):
    """Return an expression for the stress σ given a displacement field"""
    return 2.0 * μ * ufl.sym(ufl.grad(v)) + λ * ufl.tr(ufl.sym(ufl.grad(v))) * ufl.Identity(len(v))
# -

# A function space space is created and the elasticity variational
# problem defined:


ω, ρ = 300.0, 10.0
x = ufl.SpatialCoordinate(msh)
f = ufl.as_vector((ρ * ω**2 * x[0], ρ * ω**2 * x[1], 0.0))


V = fem.VectorFunctionSpace(msh, ("Lagrange", 1))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = fem.form(ufl.inner(σ(u), ufl.grad(v)) * ufl.dx)
L = fem.form(ufl.inner(f, v) * ufl.dx)

bc = fem.dirichletbc(np.zeros(3, dtype=np.float64),
                     fem.locate_dofs_topological(V, entity_dim=2, entities=facets), V=V)


A = fem.assemble_matrix(a, bcs=[bc])
A.assemble()

b = fem.assemble_vector(L)
fem.apply_lifting(b, [a], bcs=[[bc]])
# b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
fem.set_bc(b, [bc])
