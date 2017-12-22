
from dolfin.cpp.generation import RectangleMesh, BoxMesh
from dolfin.cpp import MPI
from dolfin.cpp.geometry import Point
from dolfin.cpp.mesh import CellType

def UnitSquareMesh(nx, ny):
    return RectangleMesh.create(MPI.comm_world, [Point(0.0, 0.0), Point(1.0, 1.0)],
                                [nx, ny], CellType.Type.triangle)
