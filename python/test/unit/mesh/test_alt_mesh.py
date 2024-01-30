from mpi4py import MPI

import numpy as np

import basix
import ufl
from dolfinx.cpp.fem import CoordinateElement_float64
from dolfinx.cpp.mesh import create_quad_rectangle_float64
from dolfinx.fem import CoordinateElement
from dolfinx.io import XDMFFile
from dolfinx.mesh import Mesh


def create_tensor_product_element(cell_type, degree, variant):
    """Create tensor product element."""
    family = basix.ElementFamily.P
    ref = basix.create_element(family, cell_type, degree, variant)
    factors = ref.get_tensor_product_representation()[0]
    perm = factors[1]
    dof_ordering = np.argsort(perm)
    element = basix.create_element(family, cell_type, degree, variant,
                                   dof_ordering=dof_ordering)
    return element


def test_quad_rect():
    element = create_tensor_product_element(basix.CellType.quadrilateral, 1, basix.LagrangeVariant.gll_warped)
    cmap = CoordinateElement(CoordinateElement_float64(element._e))
    _mesh = create_quad_rectangle_float64(MPI.COMM_SELF, [[0.0, 0.0], [1.0, 1.0]], [20, 20], cmap._cpp_object, None)

    e_ufl = basix.ufl._BasixElement(element)
    e_ufl = basix.ufl.blocked_element(e_ufl, shape=(2,), gdim=2)

    mesh = Mesh(_mesh, ufl.Mesh(e_ufl))

    xdmf = XDMFFile(mesh.comm, "example.xdmf", "w")
    xdmf.write_mesh(mesh)
