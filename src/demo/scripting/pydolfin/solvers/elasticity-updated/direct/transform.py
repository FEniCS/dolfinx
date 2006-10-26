from dolfin import *
import Numeric

def transform(mesh, A, b):

    vi = vertices(mesh)
    while not vi.end():

        v = Numeric.array([vi.point()[0],
                           vi.point()[1],
                           vi.point()[2]])

        v = Numeric.matrixmultiply(A, v) + b

        vi.point()[0] = v[0]
        vi.point()[1] = v[1]
        vi.point()[2] = v[2]

        vi.increment()
