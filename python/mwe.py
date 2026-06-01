import time

from mpi4py import MPI

import dolfinx.fem.petsc
import ufl

N = 30
mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, N, N, N)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

a = ufl.inner(u, v) * ufl.dx
x = ufl.SpatialCoordinate(mesh)
L = ufl.inner(ufl.sin(5 * ufl.pi * x[0]) * ufl.sin(5 * ufl.pi * x[1]), v) * ufl.dx

problem = dolfinx.fem.petsc.LinearProblem(
    a,
    L,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_reuse_preconditioner": True,
    },
    petsc_options_prefix="pp_",
)
start = time.perf_counter()
problem.solve()
end = time.perf_counter()
print(f"Time to solve: {end - start:.2e} seconds")
start_mid = time.perf_counter()
problem.solver.solve(problem.b, problem.x)
# problem.solver.getPC().setReusePreconditioner(True)
end_mid = time.perf_counter()
print(f"Time to solve (mid): {end_mid - start_mid:.2e}")
start2 = time.perf_counter()
problem.solve()
end2 = time.perf_counter()
print(f"Time to solve (second time): {end2 - start2:.2e} seconds")
