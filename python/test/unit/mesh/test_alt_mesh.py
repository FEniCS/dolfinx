from mpi4py import MPI

from dolfinx.cpp.mesh import create_quad_rectangle_float64
from dolfinx.fem import coordinate_element
from dolfinx.mesh import CellType


def test_quad_rect():

    cmap = coordinate_element(CellType.quadrilateral, 1)
    mesh = create_quad_rectangle_float64(MPI.COMM_SELF, [[0.0, 0.0], [1.0, 1.0]], [20, 20], cmap._cpp_object, None)
    assert mesh
