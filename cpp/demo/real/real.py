from ufl import FiniteElement, triangle, TrialFunction, TestFunction, dx

element = FiniteElement("Real", triangle, 0)
u = TrialFunction(element)
v = TestFunction(element)

a = u*v*dx
