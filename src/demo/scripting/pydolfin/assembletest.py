from dolfin import *
from math import *

class Source(Function):
    def eval(self, point, i):
        return point.y + 1.0

class SimpleBC(BoundaryCondition):
    def eval(self, value, point, i):
        if point.x == 0.0 or point.x == 1.0:
            value.set(0.0)
        return value
    
f = Source()
bc = SimpleBC()
#mesh = Mesh("minimal2.xml.gz")
mesh2D = UnitSquare(1, 1)
mesh3D = UnitCube(1, 1, 1)

A = Matrix()
x = Vector()
b = Vector()

import poisson3dform
import elasticityform
import stokes3dform

apoisson = poisson3dform.Poisson3DBilinearForm()
Apoisson = Matrix()
FEM_assemble(apoisson, Apoisson, mesh3D)

aelasticity = elasticityform.ElasticityBilinearForm(1.0, 1.0)
Aelasticity = Matrix()
FEM_assemble(aelasticity, Aelasticity, mesh3D)

astokes = stokes3dform.Stokes3DBilinearForm()
Astokes = Matrix()
FEM_assemble(astokes, Astokes, mesh3D)

