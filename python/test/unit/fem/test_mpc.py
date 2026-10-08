from mpi4py import MPI

import numpy as np
import pytest

from dolfinx import default_scalar_type
from dolfinx.fem import (
    Function,
    FunctionSpace,
    apply_lifting,
    assemble_vector,
    create_sparsity_pattern,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
    set_bc_diagonal,
)
from dolfinx.fem.mpc import (
    MPC,
    apply_mpc_solution,
    apply_mpc_vector,
    assemble_matrix_mpc,
    build_sparsity_pattern_mpc,
)
from dolfinx.la import InsertMode, matrix_csr
from dolfinx.mesh import create_unit_square, locate_entities_boundary
from ufl import TestFunction, TrialFunction, dx, grad, inner

try:
    from dolfinx.la.superlu_dist import superlu_dist_matrix, superlu_dist_solver
except (ImportError, RuntimeError):
    pytest.skip("dolfinx.la.superlu_dist not available", allow_module_level=True)


def test_mpc():
    mesh = create_unit_square(MPI.COMM_WORLD, 50, 50)
    facets_bc = locate_entities_boundary(
        mesh,
        dim=mesh.topology.dim - 1,
        marker=lambda x: np.isclose(x[1], 0.0) & np.isclose(x[0], 0.5, 0.5),
    )

    facets_left = locate_entities_boundary(
        mesh, dim=(mesh.topology.dim - 1), marker=lambda x: np.isclose(x[0], 0.0)
    )

    facets_right = locate_entities_boundary(
        mesh, dim=(mesh.topology.dim - 1), marker=lambda x: np.isclose(x[0], 1.0)
    )

    V = functionspace(mesh, ("Lagrange", 1))
    dofsbc = locate_dofs_topological(V=V, entity_dim=1, entities=facets_bc)

    dofsL = locate_dofs_topological(V=V, entity_dim=1, entities=facets_left)
    dofsR = locate_dofs_topological(V=V, entity_dim=1, entities=facets_right)
    coords = V.tabulate_dof_coordinates()

    ltog = V.dofmap.index_map.local_to_global(dofsR)
    globalR = np.concatenate(mesh.comm.allgather(ltog))
    globalR_coords = np.concatenate(mesh.comm.allgather(coords[dofsR]))

    coord_tol = 1e-6 if coords.dtype == np.float32 else 1e-9

    def cfun(p0, p1):
        p1t = p1 + np.array([-1.0, 0, 0.0])
        if np.linalg.norm(p0 - p1t) < coord_tol:
            return True
        return False

    # Creating mapping of left side to right side dofs
    # using local index for left, global for right.
    map_LR = {}
    for dofL in dofsL:
        xL = coords[dofL]
        for dofR, xR in zip(globalR, globalR_coords, strict=False):
            if cfun(xL, xR):
                map_LR[int(dofL)] = int(dofR)

    print(map_LR)

    # Create MPC
    local_dofs = np.array([k for k in map_LR.keys()], dtype=np.int32)
    global_dofs = [np.array([map_LR[k]], dtype=np.int64) for k in map_LR.keys()]
    global_coeffs = [np.array([1.0], dtype=np.float64) for k in map_LR.keys()]
    mpc = MPC(V, local_dofs, global_dofs, global_coeffs)
    V_new = FunctionSpace(mesh, V.ufl_element(), mpc.V)
    bc = dirichletbc(value=default_scalar_type(0), dofs=dofsbc, V=V_new)

    # Standard Poisson problem
    u = TrialFunction(V_new)
    v = TestFunction(V_new)
    a = inner(grad(u), grad(v)) * dx
    a = form(a)

    # Create SparsityPattern
    sp = create_sparsity_pattern(a)
    # Add extra MPC links to sparsity
    build_sparsity_pattern_mpc(sp, a, mpc, mpc)
    sp.finalize()

    A = matrix_csr(sp, dtype=default_scalar_type)
    assemble_matrix_mpc(mpc, A, a, [bc])
    set_bc_diagonal(A, V_new, [bc], 1.0)
    A.scatter_reverse()

    A_superlu = superlu_dist_matrix(A)
    solver = superlu_dist_solver(A_superlu)
    solver.set_option("SymmetricMode", "YES")

    f = Function(V_new)
    f.interpolate(lambda x: 50 * np.sin(np.pi * x[1] * 10) * np.exp(-30 * (x[0] - 0.05) ** 2))
    L = form(inner(f, v) * dx)

    # Assemble RHS with MPC transformation:
    #   1. scatter_rev so constrained dof entries are complete from ghost contributions
    #   2. apply_mpc_vector: b[ref] += c * b[constrained], b[constrained] = 0  (P^T step)
    #   3. scatter_rev to accumulate P^T ghost writes back to owning ranks
    #   4. apply_lifting and set Dirichlet BC values
    b = assemble_vector(L)
    b.scatter_reverse(InsertMode.add)
    apply_mpc_vector(b.array, mpc)
    b.scatter_reverse(InsertMode.add)
    apply_lifting(b.array, [a], [[bc]])
    b.scatter_reverse(InsertMode.add)
    bc.set(b.array)

    # Solve
    u = Function(V_new)
    solver.solve(b, u.x)

    # Recover constrained dof values: u[constrained] = sum c_k * u[ref_k].
    # scatter_fwd first so reference ghost dof values are current.
    u.x.scatter_forward()
    apply_mpc_solution(u.x.array, mpc)

    # Verify periodicity: u on left edge should equal u on right edge at matching y
    u_arr = u.x.array
    size_local = V_new.dofmap.index_map.size_local

    # Filter to owned DOFs only to avoid double-counting in parallel
    dofsL_owned = dofsL[dofsL < size_local]
    dofsR_owned = dofsR[dofsR < size_local]

    # Gather y-coordinates and solution values across all processes
    left_y = np.concatenate(mesh.comm.allgather(coords[dofsL_owned, 1]))
    left_u = np.concatenate(mesh.comm.allgather(u_arr[dofsL_owned]))
    right_y = np.concatenate(mesh.comm.allgather(coords[dofsR_owned, 1]))
    right_u = np.concatenate(mesh.comm.allgather(u_arr[dofsR_owned]))

    # Sort both by y so matching pairs align
    left_u = left_u[np.argsort(left_y)]
    right_u = right_u[np.argsort(right_y)]

    atol = 1e-5 if default_scalar_type in (np.float32, np.complex64) else 1e-10
    assert np.allclose(left_u, right_u, atol=atol)
