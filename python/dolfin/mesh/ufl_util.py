import ufl


def ufl_cell(mesh):
    """Returns the ufl cell of the mesh."""
    gdim = mesh.geometry().dim()
    cellname = mesh.type().description(False)
    return ufl.Cell(cellname, geometric_dimension=gdim)


def ufl_coordinate_element(mesh):
    """Return the finite element of the coordinate vector field of this
    domain.

    """
    cell = mesh.ufl_cell()
    degree = mesh.geometry().degree()
    return ufl.VectorElement("Lagrange", cell, degree,
                             dim=cell.geometric_dimension())


def ufl_domain(mesh):
    """Returns the ufl domain corresponding to the mesh."""
    # Cache object to avoid recreating it a lot
    if not hasattr(mesh, "_ufl_domain"):
        mesh._ufl_domain = ufl.Mesh(mesh.ufl_coordinate_element(),
                                    ufl_id=mesh.ufl_id(), cargo=mesh)
    return mesh._ufl_domain
