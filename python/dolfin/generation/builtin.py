
from dolfin.cpp.generation import IntervalMesh, RectangleMesh, BoxMesh
from dolfin.cpp import MPI
from dolfin.cpp.geometry import Point
from dolfin.cpp.mesh import CellType

def UnitIntervalMesh(comm, nx):
    return IntervalMesh.create(comm, nx, [0.0, 1.0])

def UnitSquareMesh(comm, nx, ny, cell_type=CellType.Type.triangle):
    return RectangleMesh.create(comm, [Point(0.0, 0.0), Point(1.0, 1.0)],
                                [nx, ny], cell_type)

def UnitCubeMesh(comm, nx, ny, nz, cell_type=CellType.Type.tetrahedron):
    return BoxMesh.create(comm, [Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0)],
                          [nx, ny, nz], cell_type)
