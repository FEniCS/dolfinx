from mpi4py import MPI

import numpy as np

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import ufl


def surface_point(t, s):
    """Map the parameters onto the strip, or onto the cylinder."""
    rho = 1 + s * np.cos(t / 2)
    z = s * np.sin(t / 2)
    return np.column_stack([rho * np.cos(t), rho * np.sin(t), z])


def ring_mesh(n, m, width, geometry_degree):
    """A Möbius strip."""
    dtype = dolfinx.default_real_type
    k = geometry_degree
    lattice = basix.create_element(
        basix.ElementFamily.P, basix.CellType.triangle, k, basix.LagrangeVariant.equispaced
    ).points
    if MPI.COMM_WORLD.rank == 0:
        ni, nj = k * n, k * m

        def node(i, j):
            if i < ni:
                return i * (nj + 1) + j
            return nj - j

        i_grid = np.repeat(np.arange(ni), nj + 1)
        j_grid = np.tile(np.arange(nj + 1), ni)
        x = surface_point(2 * np.pi * i_grid / ni, -width + 2 * width * j_grid / nj)
        cells = []
        for i in range(n):
            for j in range(m):
                a = np.array([k * i, k * j])
                b = np.array([k * (i + 1), k * j])
                c = np.array([k * i, k * (j + 1)])
                d = np.array([k * (i + 1), k * (j + 1)])
                for p0, p1, p2 in ((a, b, c), (b, d, c)):
                    ij = np.rint(
                        p0 + np.outer(lattice[:, 0], p1 - p0) + np.outer(lattice[:, 1], p2 - p0)
                    ).astype(np.int64)
                    cells.append([node(int(p), int(q)) for p, q in ij])
        cells = np.asarray(cells, dtype=np.int64)
    else:
        x = np.zeros((0, 3))
        cells = np.zeros((0, len(lattice)), dtype=np.int64)
    domain = ufl.Mesh(
        basix.ufl.element(
            "Lagrange", "triangle", k, basix.LagrangeVariant.equispaced, shape=(3,), dtype=dtype
        )
    )
    # The normal-jump diagnostic below is an interior facet integral, so
    # facets on a process boundary need their neighbouring cell.
    return dolfinx.mesh.create_mesh(
        MPI.COMM_WORLD,
        cells,
        domain,
        x.astype(dtype),
        ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
    )


n, m = 48, 8
geometry_degree = 2
degree = 2
width = 0.92
mesh = ring_mesh(n, m, width, geometry_degree)
mesh.topology.create_facet_orientations()
el_0 = basix.ufl.element("DG", mesh.topology.cell_name(), 2, dtype=dolfinx.default_real_type)
el_1 = basix.ufl.element("RT", mesh.topology.cell_name(), 3, dtype=dolfinx.default_real_type)


V = dolfinx.fem.functionspace(mesh, el_0)
Q = dolfinx.fem.functionspace(mesh, el_1)
W = ufl.MixedFunctionSpace(V, Q)

u = dolfinx.fem.Function(V)
psi = dolfinx.fem.Function(Q)
w = [u, psi]

v, tau = ufl.TestFunctions(W)

dx = ufl.Measure("dx", domain=mesh)

uD = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0))
f = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1))


alpha = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1))
phi = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1))
u0 = dolfinx.fem.Function(V)
psi0 = dolfinx.fem.Function(Q)

F = ufl.inner(ufl.div(psi), v) * dx

F -= ufl.inner(ufl.div(psi0), v) * dx
F += alpha * ufl.inner(f, v) * dx
non_lin_term = 1 / (ufl.sqrt(1 + ufl.dot(psi, psi)))
F += ufl.inner(u, ufl.div(tau)) * dx
F += phi * non_lin_term * ufl.dot(psi, tau) * dx

tol = 1e-5

petsc_options = {
    "snes_type": "newtonls",
    "snes_rtol": tol,
    "snes_atol": tol,
    "snes_max_it": 100,
    "snes_monitor": None,
    "snes_error_if_not_converged": True,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
problem = dolfinx.fem.petsc.NonlinearProblem(
    ufl.extract_blocks(F), w, bcs=[], petsc_options=petsc_options, petsc_options_prefix="LVPP_step"
)


V_out = dolfinx.fem.functionspace(mesh, ("DG", mesh.geometry.cmap.degree))
u_out = dolfinx.fem.Function(V_out)
u_out.name = "u"

bp_u = dolfinx.io.VTXWriter(mesh.comm, "u.bp", [u_out], engine="BP4")

primal_diff = u - u0
L2_squared = ufl.dot(primal_diff, primal_diff) * dx
compiled_diff = dolfinx.fem.form(L2_squared)


Q_out = dolfinx.fem.functionspace(mesh, ("DG", mesh.geometry.cmap.degree, (mesh.geometry.dim,)))
q_out = dolfinx.fem.Function(Q_out)
q_out.name = "grad(u)"
q_expr = dolfinx.fem.Expression(ufl.grad(u), Q_out.element.interpolation_points)

psi_out = dolfinx.fem.Function(Q_out)
psi_out.name = "psi"
vtx_psi = dolfinx.io.VTXWriter(
    mesh.comm,
    "psi.bp",
    [q_out, psi_out],
    engine="BP5",
)
try:
    newton_iterations = []
    for i in range(1, 100):
        alpha.value = min(2**i, 10)

        problem.solve()
        num_newton_iterations = problem.solver.getIterationNumber()
        converged = problem.solver.getConvergedReason() > 0
        newton_iterations.append(num_newton_iterations)
        ksp_reason = problem.solver.getKSP().getConvergedReason()
        print(
            f"Iteration {i}: SNES Converged={converged} ",
            f"Newton Iterations={num_newton_iterations} ",
            f"KSP Converged={ksp_reason} alpha={alpha.value}",
        )
        local_diff = dolfinx.fem.assemble_scalar(compiled_diff)
        global_diff = np.sqrt(mesh.comm.allreduce(local_diff, op=MPI.SUM))
        print(f"|delta u |= {global_diff}")
        u0.x.array[:] = u.x.array
        psi0.x.array[:] = psi.x.array

        dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)

        u_out.interpolate(u)
        bp_u.write(i)
        q_out.interpolate(q_expr)
        psi_out.interpolate(psi)
        vtx_psi.write(i)

        if global_diff < 5 * tol:
            break
finally:
    bp_u.close()
    vtx_psi.close()

print(f"Num LVPP iterations {i}, Total number of newton iterations {sum(newton_iterations)}")
print(f"{min(newton_iterations)=} and {max(newton_iterations)=}")
