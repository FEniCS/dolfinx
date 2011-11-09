from dolfin import *

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1.0)

class Obstacle(SubDomain):
    def inside(self, x, on_boundary):
        return (between(x[1], (0.5, 0.7)) and between(x[0], (0.2, 1.0)))

mesh = UnitSquare(64, 64)

left = Left()
top = Top()
right = Right()
bottom = Bottom()
obstacle = Obstacle()

domains = CellFunction("uint", mesh)
domains.set_all(0)
obstacle.mark(domains, 1)

boundaries = FacetFunction("uint", mesh)
boundaries.set_all(0)
left.mark(boundaries, 1)
top.mark(boundaries, 2)
right.mark(boundaries, 3)
bottom.mark(boundaries, 4)

V = FunctionSpace(mesh, "CG", 2)
u = TrialFunction(V)
v = TestFunction(V)

a0 = Constant(1.0)
a1 = Constant(0.01)

g1 = Expression("- 10*exp(- pow(x[1] - 0.5, 2))")
g3 = Constant("1.0")

f = Constant(1.0)

# Define new measures
dx = Measure("dx")[domains]
ds = Measure("ds")[boundaries]

F = (inner(a0*grad(u), grad(v))*dx(0) + inner(a1*grad(u), grad(v))*dx(1)
     - g1*v*ds(1) - g3*v*ds(3)
     - f*v*dx(0) - f*v*dx(1))

a, L = lhs(F), rhs(F)

bcs = [DirichletBC(V, 5.0, boundaries, 2),
       DirichletBC(V, 0.0, boundaries, 4)]

u = Function(V)
solve(a == L, u, bcs)

n = FacetNormal(mesh)
m = dot(grad(u), n)*ds(2)
value = assemble(m)
print "boundary value = ", value

m = u*dx(1)
value = assemble(m)
print "value over obstacle = ", value

plot(u, title="u")
plot(grad(u), title="Projected grad(u)")
interactive()
