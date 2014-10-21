
from dolfin import *

def test_zero():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    u1 = interpolate(Constant(1.0), V)

    bc = DirichletBC(V, 0, "on_boundary")

    # Create arbitrary matrix of size V.dim()
    #
    # Note: Identity matrix would suffice, but there doesn't seem
    # an easy way to construct it in dolfin

    v, u = TestFunction(V), TrialFunction(V)
    A0 = assemble(u*v*dx)

    # Zero rows at boundary dofs
    bc.zero(A0)

    u1_zero = Function(V)
    u1_zero.vector()[:] = A0 * u1.vector()

    boundaryIntegral = assemble(u1_zero * ds)

    assert near(boundaryIntegral, 0.0)
