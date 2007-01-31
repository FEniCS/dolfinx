import Numeric
import LinearAlgebra
from dolfin import *
import transform

#mesh = Mesh("tetmesh-1c.xml.gz")

mesh = UnitCube(6, 6, 6)

# Transform meshes
A = Numeric.zeros([3, 3], 'd')
A[0, 0] = 3.0
A[1, 1] = 3.0
A[2, 2] = 3.0
b = Numeric.zeros(3, 'd')
b[0] = 0.0
b[1] = 0.0
b[2] = 0.0
transform.transform(mesh, A, b)
