from dolfin import *

# Fine mesh
meshf = Mesh("../canal.1.xml")
facets = MeshFunction("size_t", meshf, "../facets.1.xml")

# Coarse mesh
meshc = Mesh("../canal.0.xml")

Ve = VectorElement("CG", meshf.ufl_cell(), 2)
Qe = FiniteElement("CG", meshf.ufl_cell(), 1)
Ze = MixedElement([Ve, Qe])

Zf = FunctionSpace(meshf, Ze)
Zc = FunctionSpace(meshc, Ze)

# Define the Stokes problem
z = TrialFunction(Zf)
w = TestFunction(Zf)
(u, p) = split(z)
(v, q) = split(w)

a = (
      inner(grad(u), grad(v)) * dx
    - div(v) * p * dx
    - q * div(u) * dx
    )

L = Constant(0)*q*dx

# Boundary conditions
inflow = Expression(("0.25 * (4 - x[1] * x[1])", "0.0"), degree=2)
bcs = [DirichletBC(Zf.sub(0), inflow, facets, 2),
       DirichletBC(Zf.sub(0), (0, 0), facets, 1),
       DirichletBC(Zf.sub(0), (0, 0), facets, 4)]

# Assemble
A = PETScMatrix()
b = PETScVector()
assemble_system(a, L, bcs, A_tensor=A, b_tensor=b)

# Define solver
ksp = PETScKrylovSolver()
ksp.set_operator(A)

PETScOptions.set("ksp_type", "preonly")
PETScOptions.set("pc_type", "lu")
PETScOptions.set("pc_factor_mat_solver_package", "mumps")
ksp.set_from_options()

z = Function(Zf)
ksp.solve(z.vector(), b)

File("output/velocity.pvd") << z.split(deepcopy=True)[0]
File("output/pressure.pvd") << z.split(deepcopy=True)[1]

