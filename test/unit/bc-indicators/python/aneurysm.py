
import unittest
from dolfin import *


class Test(unittest.TestCase): 
    def test(self):  

	mesh = Mesh("../../../../data/meshes/aneurysm.xml.gz")
	V = FunctionSpace(mesh, "CG", 1)
	mf =  mesh.data().mesh_function("exterior_facet_domains")

	u = TrialFunction(V)
	v = TestFunction(V)

	f = Constant(0)
	u1 = Constant(1)
	u2 = Constant(2)
	u3 = Constant(3)

	bc1 = DirichletBC(V, u1, mf, 1)
	bc2 = DirichletBC(V, u2, mf, 2)
	bc3 = DirichletBC(V, u3, mf, 3)
	bcs = [bc1, bc2, bc3]

	a = inner(grad(u), grad(v))*dx 
	L = f*v*dx  

	problem = VariationalProblem(a, L, bcs)
	u = problem.solve()

	norm = u.vector().norm("l2")
        self.assertAlmostEqual(norm,  140.112465672, 9)


if __name__ == "__main__":
    unittest.main()





