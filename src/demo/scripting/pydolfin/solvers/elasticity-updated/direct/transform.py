from dolfin import *
import Numeric

def transform(mesh, A, b):

    vi = VertexIterator(mesh)
    while not vi.end():

        v = Numeric.array([vi.coord().x,
                           vi.coord().y,
                           vi.coord().z])

        v = Numeric.matrixmultiply(A, v) + b

        vi.coord().x = v[0]
        vi.coord().y = v[1]
        vi.coord().z = v[2]

        vi.increment()
