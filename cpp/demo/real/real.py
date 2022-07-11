from ufl import FiniteElement, triangle, TrialFunction, TestFunction, dx, VectorElement, inner

element = VectorElement("Real", triangle, 0, dim=5)
u = TrialFunction(element)
v = TestFunction(element)

a = inner(u, v)*dx
