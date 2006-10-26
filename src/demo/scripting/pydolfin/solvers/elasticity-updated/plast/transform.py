from dolfin import *
import Numeric

def transform(mesh, A, b):

    geometry = mesh.geometry()

    vi = vertices(mesh)
    while not vi.end():

        v = Numeric.array([vi.point()[0],
                           vi.point()[1],
                           vi.point()[2]])

        v = Numeric.matrixmultiply(A, v) + b

        vals = geometry.x(vi.index())

        realArray_setitem(vals, 0, v[0])
        realArray_setitem(vals, 1, v[1])
        realArray_setitem(vals, 2, v[2])

        vi.increment()
