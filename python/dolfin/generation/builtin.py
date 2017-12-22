
from dolfin.cpp.generation import RectangleMesh, BoxMesh
from dolfin.cpp import MPI
from dolfin.cpp.geometry import Point
from dolfin.cpp.mesh import CellType

def UnitIntervalMesh(nx):
    return IntervalMesh.create(MPI.comm_world, nx, [Point(0.0), Point(1.0)])

def UnitSquareMesh(nx, ny, cell_type=CellType.Type.triangle):
    return RectangleMesh.create(MPI.comm_world, [Point(0.0, 0.0), Point(1.0, 1.0)],
                                [nx, ny], cell_type)

def UnitCubeMesh(nx, ny, nz, cell_type=CellType.Type.tetrahedron):
    return BoxMesh.create(MPI.comm_world, [Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0)],
                          [nx, ny, nz], cell_type)
