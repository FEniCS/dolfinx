"Demonstrating the assembly on a restricted FunctionSpace."

__author__ = "Kent-Andre Mardal (kent-and@simula.no)"
__date__ = "2009-02-9"
__copyright__ = "Copyright (C) 2008 Kent-Andre Mardal"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *
import sys

class LeftSide(SubDomain):
    def inside(self, x, on_boundary):
        return bool(x[0] < 0.5 + DOLFIN_EPS)


print "Function restrictions are not working during the parallel transition"
sys.exit()

mesh = UnitSquare(2,2)
mesh_function = MeshFunction("bool", mesh, mesh.topology().dim())
subdomain = LeftSide()
for cell in cells(mesh):
    p = cell.midpoint()
    mesh_function.values()[cell.index()] = int(subdomain.inside(p, False))
    print "c ", cell.index(), " v ",  mesh_function.values()[cell.index()]

V  = FunctionSpace(mesh, "Lagrange", 1)
fv = Function(V, cppexpr='1.0')
vv = TestFunction(V)
uv = TrialFunction(V)

m  = uv*vv*dx
L0 = fv*vv*dx
M  = assemble(m)
b  = assemble(L0)
print "size of matrix on the whole domain is ", M.size(0), "x", M.size(1)

W = V.restriction(mesh_function)
fw = Function(W, cppexpr='1.0')
vw = TestFunction(W)
uw = TrialFunction(W)

m  = uw*vw*dx
L1 = fw*vw*dx
M  = assemble(m)
b  = assemble(L1)
print "size of matrix on the smaller domain is ", M.size(0), "x", M.size(1)

m = uw*vv*dx
M = assemble(m)
file = File("M2.m")
file << M
print "size of matrix with trial functions on the smaller domain and test functions on the whole domain ", M.size(0), "x", M.size(1)

m = uv*vw*dx
M = assemble(m)
print "size of matrix with test functions on the smaller domain and trial functions on the whole domain ", M.size(0), "x", M.size(1)

# FIXME: the following is currently not working
#
#mixed = V * W
#
#(vv,vw) = TestFunctions(mixed)
#(uv,uw) = TrialFunctions(mixed)

#m = uv*vv*dx + uw*vw*dx
#L4 = fv*vv*dx + fw*vw*dx
#M = assemble(m)
#b = assemble(L4)

#file = File("M4.m"); file <<M
#file = File("b4.m"); file <<b



