import ufl
import dolfin.cpp as cpp

# Functions to extend cpp.mesh.Mesh with


def ufl_cell(self):
    return ufl.Cell(self.cell_name(),
                    geometric_dimension=self.geometry().dim())


def ufl_coordinate_element(self):
    """Return the finite element of the coordinate vector field of this
    domain.

    """
    cell = self.ufl_cell()
    degree = self.geometry().degree()
    return ufl.VectorElement("Lagrange", cell, degree,
                             dim=cell.geometric_dimension())


def ufl_domain(self):
    """Returns the ufl domain corresponding to the mesh."""
    # Cache object to avoid recreating it a lot
    if not hasattr(self, "_ufl_domain"):
        self._ufl_domain = ufl.Mesh(self.ufl_coordinate_element(),
                                    ufl_id=self.ufl_id(),
                                    cargo=self)
    return self._ufl_domain


def geometric_dimension(self):
    """Returns geometric dimension for ufl interface"""
    return self.geometry().dim()


def _repr_html_(self):
    return cpp.io.X3DOM.html(self)


# Extend cpp.mesh.Mesh class, and clean-up
cpp.mesh.Mesh.ufl_cell = ufl_cell
cpp.mesh.Mesh.ufl_coordinate_element = ufl_coordinate_element
cpp.mesh.Mesh.ufl_domain = ufl_domain
cpp.mesh.Mesh.geometric_dimension = geometric_dimension

cpp.mesh.Mesh._repr_html_ = _repr_html_

del ufl_cell, ufl_coordinate_element, ufl_domain, geometric_dimension, _repr_html_
