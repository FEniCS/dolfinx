# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Solve the Poisson and linearised elasticity equations using pyamg
#
# The demo illustrates solving the Poisson and linearised elasticity
# equations with using algebraic multigrid from
# [pyamg](https://github.com/pyamg/pyamg). It is implemented in
# {download}`demo_pyamg.py`.
#
# pyamg is not MPI-parallel, therefore this demo runs in serial only.

# +
import sys

from mpi4py import MPI

import numpy as np
import numpy.typing as npt

if MPI.COMM_WORLD.size > 1:
    print("This demo works only in serial.")
    exit(0)

try:
    import pyamg
except (ImportError, AttributeError):
    print('This demo requires pyamg, install using "pip install pyamg"')
    exit(0)


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

# -


# +
def poisson_problem(dtype: npt.DTypeLike, solver_type: str):
    """Solve a 3D Poisson problem using Ruge-Stuben algebraic multigrid.

    Args:
        dtype: Scalar type to use.
        solver_type: pyamg solver type, either "ruge_stuben" or "smoothed_aggregation"
    """

    real_type = np.real(dtype(0)).dtype
    mesh = create_box(
        comm=MPI.COMM_WORLD,
        points=[(0.0, 0.0, 0.0), (3.0, 2.0, 1.0)],
        n=[30, 20, 10],
        cell_type=CellType.tetrahedron,
        dtype=real_type,
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

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
    g = ufl.sin(5 * x[0])
    a = form(inner(grad(u), grad(v)) * dx, dtype=dtype)
    L = form(inner(f, v) * dx + inner(g, v) * ds, dtype=dtype)

    A = assemble_matrix(a, [bc]).to_scipy()
    b = assemble_vector(L)
    apply_lifting(b.array, [a], bcs=[[bc]])
    set_bc(b.array, [bc])

    # Create solution function and create AMG solver
    uh = fem.Function(V, dtype=dtype)

    if solver_type == "ruge_stuben":
        ml = pyamg.ruge_stuben_solver(A)
    elif solver_type == "smoothed_aggregation":
        ml = pyamg.smoothed_aggregation_solver(A)
    else:
        raise ValueError(f"Invalid multigrid type: {solver_type}")
    print(ml)

    # Solve linear systems
    print(f"\nSolve Poisson equation: {dtype.__name__}")
    res: list[float] = []
    tol = 1e-10 if real_type == np.float64 else 1e-6
    uh.x.array[:] = ml.solve(b.array, tol=tol, residuals=res, accel="cg")
    for i, q in enumerate(res):
        print(f"Convergence history: iteration {i}, residual= {q}")

    with io.XDMFFile(mesh.comm, f"out_pyamg/poisson_{dtype.__name__}.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(uh)


# -


# +
def nullspace_elasticty(Q: fem.FunctionSpace) -> list[np.ndarray]:
    """Create the elasticity (near)nulspace.

    Args:
        Q: Displacement field function space.

    Returns:
        Nullspace for the unconstrained problem.
    """
    B = np.zeros((Q.dofmap.index_map.size_local * Q.dofmap.bs, 6))

    # Get dof indices for each subspace (x, y and z dofs)
    dofs = [Q.sub(i).dofmap.list.flatten() for i in range(3)]

    # Set the three translational rigid body modes
    for i in range(3):
        B[dofs[i], i] = 1.0

    # Set the three rotational rigid body modes
    x = Q.tabulate_dof_coordinates()
    dofs_block = Q.dofmap.list.flatten()
    x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
    B[dofs[0], 3] = -x1
    B[dofs[1], 3] = x0
    B[dofs[0], 4] = x2
    B[dofs[2], 4] = -x0
    B[dofs[2], 5] = x1
    B[dofs[1], 5] = -x2
    return B


# -


# +
def elasticity_problem(dtype):
    """Solve a 3D linearised elasticity problem using a smoothed
    aggregation algebraic multigrid method.

    Args:
        dtype: Scalar type to use.
    """
    mesh = create_box(
        comm=MPI.COMM_WORLD,
        points=[(0.0, 0.0, 0.0), (3.0, 2.0, 1.0)],
        n=[30, 20, 10],
        cell_type=CellType.tetrahedron,
        dtype=dtype,
    )

    facets = locate_entities_boundary(
        mesh,
        dim=(mesh.topology.dim - 1),
        marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 3.0),
    )

    ω, ρ = 300.0, 10.0
    x = ufl.SpatialCoordinate(mesh)
    f = ufl.as_vector((ρ * ω**2 * x[0], ρ * ω**2 * x[1], 0.0))

    # Define the elasticity parameters and create a function that
    # computes an expression for the stress given a displacement field.
    E = 1.0e9
    ν = 0.3
    μ = E / (2.0 * (1.0 + ν))
    λ = E * ν / ((1.0 + ν) * (1.0 - 2.0 * ν))

    def σ(v):
        """Return an expression for the stress σ given a displacement field"""
        return 2.0 * μ * ufl.sym(grad(v)) + λ * ufl.tr(ufl.sym(grad(v))) * ufl.Identity(len(v))

    V = functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = form(inner(σ(u), grad(v)) * dx, dtype=dtype)
    L = form(inner(f, v) * dx, dtype=dtype)

    tdim = mesh.topology.dim
    dofs = locate_dofs_topological(V=V, entity_dim=tdim - 1, entities=facets)
    bc = dirichletbc(np.zeros(3, dtype=dtype), dofs, V=V)

    A = assemble_matrix(a, bcs=[bc]).to_scipy()
    b = assemble_vector(L)
    apply_lifting(b.array, [a], bcs=[[bc]])
    set_bc(b.array, [bc])

    uh = fem.Function(V, dtype=dtype)
    B = nullspace_elasticty(V)
    ml = pyamg.smoothed_aggregation_solver(A, B=B)
    print(ml)

    print(f"\nLinearised elasticity: {dtype.__name__}")
    res_e: list[float] = []
    tol = 1e-10 if dtype == np.float64 else 1e-6
    uh.x.array[:] = ml.solve(b.array, tol=tol, residuals=res_e, accel="cg")
    for i, q in enumerate(res_e):
        print(f"Convergence history: iteration {i}, residual= {q}")

    with io.XDMFFile(mesh.comm, f"out_pyamg/elasticity_{dtype.__name__}.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(uh)


# -


# Solve Poission problem with different scalar types
poisson_problem(np.float32, "ruge_stuben")
poisson_problem(np.float64, "ruge_stuben")

# For complex, pyamg requires smoothed aggregation multigrid
if not sys.platform.startswith("win32"):
    poisson_problem(np.complex64, "smoothed_aggregation")
    poisson_problem(np.complex128, "smoothed_aggregation")

# Solve elasticity problem with different scalar types
elasticity_problem(np.float32)
elasticity_problem(np.float64)
# -
