from dolfin import *

V = FunctionSpace(UnitSquare(32, 32), "CG", 1)
plot(VariationalProblem((grad(TestFunction(V)), grad(TrialFunction(V))) + (TestFunction(V), TrialFunction(V)), (TestFunction(V), Expression("sin(x[0])*sin(x[1])"))).solve(), interactive=True)
