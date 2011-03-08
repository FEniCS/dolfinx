from dolfin import *

V = FunctionSpace(UnitSquare(32, 32), "CG", 1)
plot(VariationalProblem((grad(TrialFunction(V)), grad(TestFunction(V))) + (TrialFunction(V), TestFunction(V)), (TestFunction(V), Expression("sin(x[0])*sin(x[1])"))).solve(), interactive=True)
