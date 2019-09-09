
import ufl
import dolfin


def test_interpolation():
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 5, 5)
    vP1 = dolfin.VectorFunctionSpace(mesh, ("P", 1))

    u1 = dolfin.Function(vP1)
    u1.vector.set(1.0)

    k1 = dolfin.Constant(mesh, [[1.0, 2.0], [3.0, 4.0]])
    x = ufl.SpatialCoordinate(mesh)
    expr = ufl.inner(x, u1) * ufl.dot(k1, u1)

    f = dolfin.Function(vP1)
    dolfin.fem.interpolation.compiled_interpolation(expr, vP1, f)
