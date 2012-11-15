"Demonstrating the assembly on a restricted FunctionSpace."

# Copyright (C) 2008 Kent-Andre Mardal
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2009-02-09
# Last changed: 2012-11-12

from dolfin import *
import sys

class LeftSide(SubDomain):
    def inside(self, x, on_boundary):
        return bool(x[0] < 0.5 + DOLFIN_EPS)


print "Function restrictions are not working during the parallel transition"
sys.exit()

mesh = UnitSquareMesh(2,2)
mesh_function = MeshFunction("bool", mesh, mesh.topology().dim())
subdomain = LeftSide()
for cell in cells(mesh):
    p = cell.midpoint()
    mesh_function.array()[cell.index()] = int(subdomain.inside(p, False))
    print "c ", cell.index(), " v ",  mesh_function.array()[cell.index()]

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



