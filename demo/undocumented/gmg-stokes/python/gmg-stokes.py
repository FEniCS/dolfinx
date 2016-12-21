from dolfin import *
from petsc4py import PETSc

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
solver = PETScKrylovSolver()
solver.set_operator(A)

dm_collection = PETScDMCollection([Zc, Zf])
solver.set_dm(dm_collection)

PETScOptions.set("ksp_type", "preonly")
PETScOptions.set("pc_type", "lu")
PETScOptions.set("pc_factor_mat_solver_package", "mumps")
solver.set_from_options()

ksp = solver.ksp()
pc  = ksp.pc
if pc.type == "fieldsplit":
    # Assemble the mass matrix and specify it as the approximation to the Schur complement
    dm  = pc.getDM()
    (names, ises, dms) = dm.createFieldDecomposition() # fetch subdm corresponding to pressure space

    M = PETScMatrix()
    assemble(-inner(p, q)*dx, tensor=M)
    mass = M.mat().getSubMatrix(ises[1], ises[1])

    ksp_mass = PETSc.KSP().create()
    ksp_mass.setDM(dms[1])
    ksp_mass.setDMActive(False) # don't try to build the operator from the DM
    ksp_mass.setOperators(mass) # solve the mass matrix
    ksp_mass.setOptionsPrefix("mass_")
    ksp_mass.setFromOptions()

    class SchurApproxInv(object):
        def mult(self, mat, x, y):
            ksp_mass.solve(x, y)
    schurpc = PETSc.Mat()
    schurpc.createPython(mass.getSizes(), SchurApproxInv())
    schurpc.setUp()

    pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.USER, schurpc)

z = Function(Zf)
solver.solve(z.vector(), b)

File("output/velocity.pvd") << z.split(deepcopy=True)[0]
File("output/pressure.pvd") << z.split(deepcopy=True)[1]

