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
# This demo . . .

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import ufl
from dolfinx import cpp as _cpp
from dolfinx import fem, la, mesh, plot
from dolfinx.fem.assemble import pack_coefficients, pack_constants

from mpi4py import MPI

# Create a mesh and function space
msh = mesh.create_rectangle(comm=MPI.COMM_SELF,
                            points=((0.0, 0.0), (2.0, 1.0)), n=(32, 16),
                            cell_type=mesh.CellType.triangle)
V = fem.FunctionSpace(msh, ("Lagrange", 1))


# Define L2 projection problem
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
f = ufl.sin(2 * np.pi * x[0])
fc = f + ufl.sin(2 * np.pi * x[0]) + ufl.sin(4 * np.pi * x[1]) * 1j
a = ufl.inner(u, v) * ufl.dx
L = ufl.inner(f, v) * ufl.dx


def project(dtype=np.float32):
    """Solve the simple L2 projection problem"""

    a0 = fem.form(a, dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        L0 = fem.form(ufl.replace(L, {f: fc}), dtype=dtype)
    else:
        L0 = fem.form(L, dtype=dtype)

    # Create a sparsity pattern for initialising the sparse matrix
    # NOTE: the sparsity pattern does not depend of the dtype and could
    # be re-used
    sp = fem.create_sparsity_pattern(a0)
    sp.assemble()

    # Create a sparse matrix and assemble
    A = la.matrix_csr(sp, dtype=dtype)
    _cpp.fem.assemble_matrix(A, a0, [])
    A.finalize()

    # Create a vector and assemble
    b = la.vector(L0.function_spaces[0].dofmap.index_map, dtype=dtype)
    _cpp.fem.assemble_vector(b.array, L0, pack_constants(L0), pack_coefficients(L0))

    # Create a Scipy sparse matrix that shares data with A
    As = scipy.sparse.csr_matrix((A.data, A.indices, A.indptr))

    # Solve the variational problem and return the solution
    uh = fem.Function(V, dtype=dtype)
    uh.x.array[:] = scipy.sparse.linalg.spsolve(As, b.array)
    return uh


def display(u, filter=np.real):
    """PLot the solution using pyvista"""
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
