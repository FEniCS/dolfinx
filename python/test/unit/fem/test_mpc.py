from mpi4py import MPI

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

import dolfinx.cpp
from dolfinx.fem import (
    Function,
    create_sparsity_pattern,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.la import matrix_csr
from dolfinx.mesh import create_unit_square, locate_entities_boundary
from ufl import TestFunction, TrialFunction, dx, grad, inner


@pytest.mark.skip_in_parallel
def test_mpc():
    mesh = create_unit_square(MPI.COMM_WORLD, 50, 10)
    facets_bc = locate_entities_boundary(
        mesh,
        dim=mesh.topology.dim - 1,
        marker=lambda x: np.isclose(x[1], 0.0) & np.isclose(x[0], 0.1, 0.25),
    )

    facets_left = locate_entities_boundary(
        mesh, dim=(mesh.topology.dim - 1), marker=lambda x: np.isclose(x[0], 0.0)
    )

    facets_right = locate_entities_boundary(
        mesh, dim=(mesh.topology.dim - 1), marker=lambda x: np.isclose(x[0], 1.0)
    )

    V = functionspace(mesh, ("Lagrange", 1))
    dofsbc = locate_dofs_topological(V=V, entity_dim=1, entities=facets_bc)
    bc = dirichletbc(value=0.0, dofs=dofsbc, V=V)
    assert len(dofsbc) > 0

    dofsL = locate_dofs_topological(V=V, entity_dim=1, entities=facets_left)
    dofsR = locate_dofs_topological(V=V, entity_dim=1, entities=facets_right)
    coords = V.tabulate_dof_coordinates()

    ltog = V.dofmap.index_map.local_to_global(dofsR)
    globalR = mesh.comm.allgather(ltog)
    globalR_coords = mesh.comm.allgather(coords[dofsR])

    def cfun(p0, p1):
        p1t = p1 + np.array([-1.0, 0, 0.0])
        if np.linalg.norm(p0 - p1t) < 1e-9:
            return True
        return False

    # Creating mapping of left side to right side dofs
    # using local index for left, global for right.
    map_LR = {}
    for dofL in dofsL:
        xL = coords[dofL]
        for p in range(len(globalR)):
            for dofR, xR in zip(globalR[p], globalR_coords[p]):
                if cfun(xL, xR):
                    map_LR[int(dofL)] = int(dofR)

    print(map_LR)

    # Create MPC
    local_dofs = np.array([k for k in map_LR.keys()], dtype=np.int32)
    global_dofs = [np.array([map_LR[k]], dtype=np.int64) for k in map_LR.keys()]
    global_coeffs = [np.array([1.0], dtype=np.float64) for k in map_LR.keys()]
    mpc = dolfinx.cpp.fem.MPC_float64(V._cpp_object, local_dofs, global_dofs, global_coeffs)

    # Standard Poisson problem
    u = TestFunction(V)
    v = TrialFunction(V)
    a = inner(grad(u), grad(v)) * dx
    a = form(a)

    # Create SparsityPattern
    sp = create_sparsity_pattern(a)
    # Add extra sparsity for MPC connections
    for cell in mpc.cells():
        cell_dofs = np.array(V.dofmap.cell_dofs(cell))
        new_rc = np.array(mpc.modified_dofs(cell_dofs))
        sp.insert(new_rc, new_rc)

    # Add extra sparsity for constraint
    for dof in range(V.dofmap.index_map.size_local):
        c = mpc.constraint(dof)
        if len(c[0]) > 0:
            row_col = np.array([dof] + c[0])
            sp.insert(row_col, row_col)
    sp.finalize()

    A = matrix_csr(sp)

    dolfinx.cpp.fem.assemble_matrix_mpc(mpc, A._cpp_object, a._cpp_object, [bc._cpp_object])
    dolfinx.cpp.fem.insert_diagonal(A._cpp_object, a.function_spaces[0], [bc._cpp_object], 1.0)

    # Convert to scipy
    A = A.to_scipy()

    # Fake RHS, setting constraint b_i to zero
    b = np.ones(A.shape[1])
    for i in range(A.shape[0]):
        c = mpc.constraint(i)
        if len(c[0]) > 0:
            b[i] = 0.0

    # Solve
    u = Function(V)
    u.x.array[:] = spsolve(A, b)
    result = u.x.array

    # Check MPC has worked
    for k in map_LR:
        assert np.isclose(result[k], result[map_LR[k]])
