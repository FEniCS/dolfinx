from dolfin import *

class Domain (SubDomain):

    def inside(self, x, on_boundary):
        return (x[0] > 0.25 - DOLFIN_EPS and
                x[0] < 0.75 + DOLFIN_EPS and
                x[1] > 0.25 - DOLFIN_EPS and
                x[1] < 0.75 + DOLFIN_EPS)

domain = Domain()

mesh = UnitSquareMesh(32, 32)
restriction = Restriction(mesh, domain)
V = FunctionSpace(restriction, "CG", 1)
