# Verification of projection between non-matching meshes
#
# We have that |f - Pf|_L2 <= |f - pif|_L2,
# where Pf is the L2 projection of f and pif is the interpolant of f
#
# We use this to verify the projection: we compare the projection error with
# the interpolation error. The projection error should be comparable in size,
# or at least not larger.

from dolfin import *
from math import *
import ffc.compiler.compiler as ffc
from interpolation import *

# Define right-hand side
class MyFunction(Function):
    def eval(self, point, i):
        return sin(10.0 * pow(point[0], 3.0) * point[1])
        #return 0.1 * pow(3.0 * point[0], 3.0) * point[1]

KP1 = P1tri()
K = ffc.FiniteElement("Lagrange", "triangle", 1)

# Meshes
meshA = Mesh("finer.xml")
meshB = Mesh("coarse.xml")
meshC = UnitSquare(200, 200)


# Define function to interpolate/project
g = MyFunction()
f = interpolate(g, KP1, meshA)
#f = project(g, K, meshA)
fC = interpolate(f, KP1, meshC)
fN = Function(f, meshB)

# Project on space B
Pf = project(fN, K, meshB)

# Interpolate on space B
pif = interpolate(fN, KP1, meshB)

# Interpolate to reference mesh
Pf_C = interpolate(Pf, KP1, meshC)
pif_C = interpolate(pif, KP1, meshC)

# Compute error wrt. reference mesh
pif_error = Function(meshC, KP1)
pif_error.vector().copy(fC.vector(), 0, 0, pif_error.vector().size())
pif_error.vector().axpy(-1.0, pif_C.vector())

Pf_error = Function(meshC, KP1)
Pf_error.vector().copy(fC.vector(), 0, 0, pif_error.vector().size())
Pf_error.vector().axpy(-1.0, Pf_C.vector())

difference = Function(meshC, KP1)
difference.vector().copy(Pf_C.vector(), 0, 0, Pf_C.vector().size())
difference.vector().axpy(-1.0, pif_C.vector())

Pf_error_C = Function(Pf_error, meshC)
pif_error_C = Function(pif_error, meshC)

L2forms = import_formfile("L2Norm.form")

M = L2forms.L2NormFunctional()

Pf_L2norm = sqrt(M(Pf_error_C, meshC))
pif_L2norm = sqrt(M(pif_error_C, meshC))

Pf_maxnorm = Pf_error.vector().norm(Vector.linf)
pif_maxnorm = pif_error.vector().norm(Vector.linf)

# Save solution to file in VTK format
file = File("function.pvd")
file << f

file = File("projection.pvd")
file << Pf

file = File("interpolation.pvd")
file << pif

file = File("projection_fine.pvd")
file << Pf_C

file = File("interpolation_fine.pvd")
file << pif_C

file = File("projection_error.pvd")
file << Pf_error

file = File("interpolation_error.pvd")
file << pif_error

file = File("difference.pvd")
file << difference

print "cells meshA: ", meshA.numCells()
print "cells meshB: ", meshB.numCells()
print "cells meshC: ", meshC.numCells()

print "Pf_L2norm: ", Pf_L2norm
print "pif_L2norm: ", pif_L2norm

print "Pf_maxnorm: ", Pf_maxnorm
print "pif_maxnorm: ", pif_maxnorm
