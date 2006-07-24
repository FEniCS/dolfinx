from dolfin import *

# Define right-hand side
class Source(Function):
    def eval(self, point, i):
        return point.y + 1.0

# Define boundary condition
class SimpleBC(BoundaryCondition):
    def eval(self, value, point, i):
        if point.x == 0.0 or point.x == 1.0:
            value.set(0.0)

f = Source()
bc = SimpleBC()

# Create a mesh of the unit square
mesh = UnitSquare(10, 10)

# Import forms (compiled just-in-time with FFC)
forms = import_formfile("Poisson.form")

a = forms.PoissonBilinearForm()
L = forms.PoissonLinearForm(f)

# Assemble linear system
A = PETScSparseMatrix()
b = PETScVector()

#A = uBlasSparseMatrix()
#b = DenseVector()

assemble(a, L, A, b, mesh, bc)

# Solve linear system
x = PETScVector()
solver = PETScKrylovSolver()

#x = DenseVector()
#solver = uBlasKrylovSolver()

solver.solve(A, x, b)

# Define a function from computed degrees of freedom
trial_element = forms.PoissonBilinearFormTrialElement()
u = Function(x, mesh, trial_element)

# Save solution to file in VTK format
file = File("poisson.pvd")
file << u
