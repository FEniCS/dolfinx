from dolfin import *

comm_world = mpi_comm_world()

def test_nonlinear_variational_solver_parallel():
    if comm_world.rank == 0:
        mesh = UnitIntervalMesh(mpi_comm_self(), 2)
        V = FunctionSpace(mesh, "CG", 1)

        f = Constant(1)

        u = Function(V)
        v = TestFunction(V)

        F = inner(u, v)*dx - inner(f, v)*dx
        solve(F == 0, u) # check this doesn't deadlock
