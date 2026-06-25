from mpi4py import MPI

import numpy as np

import dolfinx.cpp
from dolfinx.fem import (
    Function,
    FunctionSpace,
    create_sparsity_pattern,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.la import matrix_csr
from dolfinx.la.superlu_dist import superlu_dist_matrix, superlu_dist_solver
from dolfinx.mesh import create_unit_square, locate_entities_boundary
from ufl import TestFunction, TrialFunction, dx, grad, inner


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
        for dofR, xR in zip(globalR, globalR_coords):
            if cfun(xL, xR):
                map_LR[int(dofL)] = int(dofR)

    print(map_LR)

    # Create MPC
    local_dofs = np.array([k for k in map_LR.keys()], dtype=np.int32)
    global_dofs = [np.array([map_LR[k]], dtype=np.int64) for k in map_LR.keys()]
    global_coeffs = [np.array([1.0], dtype=np.float64) for k in map_LR.keys()]
    mpc = dolfinx.cpp.fem.MPC_float64(V._cpp_object, local_dofs, global_dofs, global_coeffs)
    V_new = FunctionSpace(mesh, V.ufl_element(), mpc.V())
    bc = dirichletbc(value=0.0, dofs=dofsbc, V=V_new)

    # Standard Poisson problem
    u = TestFunction(V_new)
    v = TrialFunction(V_new)
    a = inner(grad(u), grad(v)) * dx
    a = form(a)

    # Create SparsityPattern
    sp = create_sparsity_pattern(a)
    # Add extra sparsity for MPC connections
    for cell in mpc.cells():
        cell_dofs = np.array(V_new.dofmap.cell_dofs(cell))
        new_rc = np.array(mpc.modified_dofs(cell_dofs))
        sp.insert(new_rc, new_rc)

    # Add extra sparsity for constraint
    for dof in range(V_new.dofmap.index_map.size_local):
        c = mpc.constraint(dof)
        if len(c[0]) > 0:
            row_col = np.array([dof] + c[0])
            sp.insert(row_col, row_col)
    sp.finalize()

    A = matrix_csr(sp)
    dolfinx.cpp.fem.assemble_matrix_mpc(mpc, A._cpp_object, a._cpp_object, [bc._cpp_object])
    dolfinx.cpp.fem.insert_diagonal(A._cpp_object, a.function_spaces[0], [bc._cpp_object], 1.0)
    A.scatter_reverse()

    A_superlu = superlu_dist_matrix(A)
    solver = superlu_dist_solver(A_superlu)
    solver.set_option("SymmetricMode", "YES")

    f = Function(V_new)
    f.interpolate(lambda x: 50 * np.sin(np.pi * x[1] * 10) * np.exp(-30 * (x[0] - 0.05) ** 2))
    L = form(inner(f, v) * dx)

    b = dolfinx.fem.assemble_vector(L)
    dolfinx.fem.apply_lifting(b.array, [a], [[bc]])
    b.scatter_reverse(dolfinx.la.InsertMode.add)
    bc.set(b.array)
    for i in range(V_new.dofmap.index_map.size_local):
        c = mpc.constraint(i)
        if len(c[0]) > 0:
            b.array[i] = 0.0

    # Solve
    u = Function(V_new)
    solver.solve(b, u.x)

    # xdmf = dolfinx.io.XDMFFile(mesh.comm, "demo.xdmf", "w")
    # xdmf.write_mesh(mesh)
    # u.name = "u"
    # xdmf.write_function(u)

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

    assert np.allclose(left_u, right_u, atol=1e-10)
