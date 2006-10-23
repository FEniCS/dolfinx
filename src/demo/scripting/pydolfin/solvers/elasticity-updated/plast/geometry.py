import Numeric
import LinearAlgebra
from dolfin import *
import transform

#mesh = Mesh("cell.xml.gz")

mesh = UnitCube(2, 2, 2)

# Transform meshes
A = Numeric.zeros([3, 3], 'd')
A[0, 0] = 1.0
A[1, 1] = 1.0
A[2, 2] = 1.0
b = Numeric.zeros(3, 'd')
b[0] = 0.0
b[1] = 0.0
b[2] = 0.0
transform.transform(mesh, A, b)
