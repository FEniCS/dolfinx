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
# This demo shows
# - How to solve problems using different scalar types, .e.g. single or
#   double precision, or complex numbers
# - Interfacing with [SciPy](https://scipy.org/) sparse linear algebra
#   functionality

# +
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import ufl
from dolfinx import fem, mesh, plot

from mpi4py import MPI
# -

# SciPy solvers do no support MPI, so all computation will be performed
# on a single MPI rank

# +
comm = MPI.COMM_SELF
# -

# Create a mesh and function space

msh = mesh.create_rectangle(comm=comm, points=((0.0, 0.0), (2.0, 1.0)), n=(32, 16),
                            cell_type=mesh.CellType.triangle)
V = fem.FunctionSpace(msh, ("Lagrange", 1))

# Define an L2 projection problem
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
fr = ufl.sin(2 * np.pi * x[0])
fc = ufl.sin(2 * np.pi * x[0]) + ufl.sin(4 * np.pi * x[1]) * 1j
a = ufl.inner(u, v) * ufl.dx
L = ufl.inner(fr + fc, v) * ufl.dx


def project(dtype=np.float32):
    """Solve the simple L2 projection problem"""

    # Process forms. This will compiler the forms for the requested type,
    a0 = fem.form(a, dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        L0 = fem.form(L, dtype=dtype)
    else:
        L0 = fem.form(ufl.replace(L, {fc: 0}), dtype=dtype)

    # Assemble forms
    A = fem.assemble_matrix(a0)
    A.finalize()
    b = fem.assemble_vector(L0)

    # Create a Scipy sparse matrix that shares data with A
    As = scipy.sparse.csr_matrix((A.data, A.indices, A.indptr))

    # Solve the variational problem and return the solution
    uh = fem.Function(V, dtype=dtype)
    uh.x.array[:] = scipy.sparse.linalg.spsolve(As, b.array)
    return uh


def display(u, filter=np.real):
    """Plot the solution using pyvista"""
    try:
        import pyvista
        cells, types, x = plot.create_vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(cells, types, x)
        grid.point_data["u"] = filter(u.x.array)
        grid.set_active_scalars("u")
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_edges=True)
        warped = grid.warp_by_scalar()
        plotter.add_mesh(warped)
        plotter.show()
    except ModuleNotFoundError:
        print("'pyvista' is required to visualise the solution")


# Solve the projection problem using different scalar types
uh = project(dtype=np.float32)
uh = project(dtype=np.float64)
uh = project(dtype=np.complex128)

# Display the last computed solution
display(uh, np.real)
display(uh, np.imag)
