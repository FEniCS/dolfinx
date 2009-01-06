
from dolfin import *

class LeftSide(SubDomain):
    def inside(self, x, on_boundary):
        return bool(x[0] < 0.5 + DOLFIN_EPS)


mesh = UnitSquare(2,2)
mesh.disp()
mesh_function = MeshFunction("bool", mesh, mesh.topology().dim())
subdomain = LeftSide()
for cell in cells(mesh): 
    p = cell.midpoint()
    mesh_function.values()[cell.index()] = int(subdomain.inside(p, False))
    print "c ", cell.index(), " v ",  mesh_function.values()[cell.index()] 

V = FunctionSpace(mesh, "Lagrange", 1)
fv = Function(V, cppexpr='1.0')
vv = TestFunction(V)
uv = TrialFunction(V)

m = uv*vv*dx
L0 = fv*vv*dx 
M = assemble(m)
b = assemble(L0)
file = File("M0.m"); file <<M
file = File("b0.m"); file <<b


#W = V.restriction(mesh_function) 
W = V 
fw = Function(W, cppexpr='1.0')
vw = TestFunction(W)
uw = TrialFunction(W)

m = uw*vw*dx
L1 = fw*vw*dx 

M = assemble(m)
b = assemble(L1)

file = File("M1.m"); file <<M
file = File("b1.m"); file <<b

mixed = MixedFunctionSpace([V,W]) 
#fm = Function(mixed, cppexpr='1.0')
vm = TestFunction(mixed)
um = TrialFunction(mixed)

print dir(vm)
#fv, fw = fm.split()
vv, vw = vm.split()
uv, uw = um.split()


m = uv*vv*dx + uw*vw*dx
#L2 = fv*vv*dx + fw*vw*dx 
M = assemble(m)
#b = assemble(L2)

file = File("M2.m"); file <<M
file = File("b2.m"); file <<b



