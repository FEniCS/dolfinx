# -*- coding: utf-8 -*-
"""This module defines some special functions (originally defined in
SpecialFunctions.h).

"""

# Copyright (C) 2008-2014 Anders Logg
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

__all__ = ["MeshCoordinates", "FacetArea", "FacetNormal",
           "CellVolume", "SpatialCoordinate", "CellNormal",
           "CellDiameter", "CellSize", "Circumradius",
           "MinCellEdgeLength", "MaxCellEdgeLength",
           "MinFacetEdgeLength", "MaxFacetEdgeLength"]

import ufl
import dolfin.cpp as cpp
from dolfin.function.expression import BaseExpression


def _mesh2domain(mesh):
    "Deprecation mechanism for symbolic geometry."

    if isinstance(mesh, ufl.cell.AbstractCell):
        raise TypeError("Cannot construct geometry from a Cell. Pass the mesh instead, for example use FacetNormal(mesh) instead of FacetNormal(triangle) or triangle.n")
    return mesh.ufl_domain()


class MeshCoordinates(BaseExpression):
    def __init__(self, mesh):
        """Create function that evaluates to the mesh coordinates at each
        vertex.

        """
        # Initialize C++ part
        self._cpp_object = cpp.function.MeshCoordinates(mesh)

        # Initialize UFL part
        ufl_element = mesh.ufl_domain().ufl_coordinate_element()
        if ufl_element.family() != "Lagrange" or ufl_element.degree() != 1:
            raise RuntimeError("MeshCoordinates only supports affine meshes")
        super().__init__(element=ufl_element, domain=mesh.ufl_domain())


class FacetArea(BaseExpression):
    def __init__(self, mesh):
        """Create function that evaluates to the facet area/length on each
        facet.

        *Arguments*
            mesh
                a :py:class:`Mesh <dolfin.cpp.Mesh>`.

        *Example of usage*

            .. code-block:: python

                mesh = UnitSquare(4,4)
                fa = FacetArea(mesh)

        """

        # Initialize C++ part
        self._cpp_object = cpp.function.FacetArea(mesh)

        # Initialize UFL part
        # NB! This is defined as a piecewise constant function for
        # each cell, not for each facet!
        ufl_element = ufl.FiniteElement("Discontinuous Lagrange", mesh.ufl_cell(), 0)
        super().__init__(domain=mesh.ufl_domain(), element=ufl_element, label="FacetArea")


# Simple definition of FacetNormal via UFL
def FacetNormal(mesh):
    """Return symbolic facet normal for given mesh.

    *Arguments*
        mesh
            a :py:class:`Mesh <dolfin.cpp.Mesh>`.


    *Example of usage*

        .. code-block:: python

            mesh = UnitSquare(4,4)
            n = FacetNormal(mesh)

    """

    return ufl.FacetNormal(_mesh2domain(mesh))


# Simple definition of CellSize via UFL
def CellDiameter(mesh):
    """Return function cell diameter for given mesh.

    Note that diameter of cell :math:`K` is defined as
    :math:`\sup_{\mathbf{x,y}\in K} |\mathbf{x-y}|`.

    *Arguments*
        mesh
            a :py:class:`Mesh <dolfin.cpp.Mesh>`.

    *Example of usage*

        .. code-block:: python

            mesh = UnitSquare(4,4)
            h = CellDiameter(mesh)

    """

    return ufl.CellDiameter(_mesh2domain(mesh))


# Simple definition of CellSize via UFL
def CellSize(mesh):
    """Return function twice the cell circumradius for given
    mesh.

    This does not generalize to quadrilateral/hexahedral cells
    and the function name is misleading. The function is hence
    deprecated. Use ``CellDiameter`` (or ``2*Circumradius`` if
    you really want that).

    *Arguments*
        mesh
            a :py:class:`Mesh <dolfin.cpp.Mesh>`.

    *Example of usage*

        .. code-block:: python

            mesh = UnitSquare(4,4)
            D = CellSize(mesh)

    """

    cpp.deprecation("\"CellSize\"",
                    "2017.2.0",
                    "Use \"CellDiameter\" or \"2*Circumradius\".")
    return 2.0*ufl.Circumradius(_mesh2domain(mesh))


# Simple definition of CellVolume via UFL
def CellVolume(mesh):
    """Return symbolic cell volume for given mesh.

    *Arguments*
        mesh
            a :py:class:`Mesh <dolfin.cpp.Mesh>`.

    *Example of usage*

        .. code-block:: python

            mesh = UnitSquare(4,4)
            vol = CellVolume(mesh)

    """

    return ufl.CellVolume(_mesh2domain(mesh))


# Simple definition of SpatialCoordinate via UFL
def SpatialCoordinate(mesh):
    """Return symbolic physical coordinates for given mesh.

    *Arguments*
        mesh
            a :py:class:`Mesh <dolfin.cpp.Mesh>`.

    *Example of usage*

        .. code-block:: python

            mesh = UnitSquare(4,4)
            x = SpatialCoordinate(mesh)

    """

    return ufl.SpatialCoordinate(_mesh2domain(mesh))


# Simple definition of CellNormal via UFL
def CellNormal(mesh):
    """Return symbolic cell normal for given manifold mesh.

    *Arguments*
        mesh
            a :py:class:`Mesh <dolfin.cpp.Mesh>`.

    *Example of usage*

        .. code-block:: python

            mesh = UnitSquare(4,4)
            n = CellNormal(mesh)

    """

    return ufl.CellNormal(_mesh2domain(mesh))


# Simple definition of Circumradius via UFL
def Circumradius(mesh):
    """Return symbolic cell circumradius for given mesh.

    *Arguments*
        mesh
            a :py:class:`Mesh <dolfin.cpp.Mesh>`.

    *Example of usage*

        .. code-block:: python

            mesh = UnitSquare(4,4)
            R = Circumradius(mesh)

    """

    return ufl.Circumradius(_mesh2domain(mesh))

# Simple definition of MinCellEdgeLength via UFL
def MinCellEdgeLength(mesh):
    """Return symbolic minimum cell edge length of a cell
    for given mesh.

    *Arguments*
        mesh
            a :py:class:`Mesh <dolfin.cpp.Mesh>`.

    *Example of usage*

        .. code-block:: python

            mesh = UnitSquare(4,4)
            mince = MinCellEdgeLength(mesh)

    """

    return ufl.MinCellEdgeLength(_mesh2domain(mesh))

# Simple definition of MaxCellEdgeLength via UFL
def MaxCellEdgeLength(mesh):
    """Return symbolic maximum cell edge length of a cell
    for given mesh.

    *Arguments*
        mesh
            a :py:class:`Mesh <dolfin.cpp.Mesh>`.

    *Example of usage*

        .. code-block:: python

            mesh = UnitSquare(4,4)
            maxce = MaxCellEdgeLength(mesh)

    """

    return ufl.MaxCellEdgeLength(_mesh2domain(mesh))

# Simple definition of MinFacetEdgeLength via UFL
def MinFacetEdgeLength(mesh):
    """Return symbolic minimum facet edge length of a cell
    for given mesh.

    *Arguments*
        mesh
            a :py:class:`Mesh <dolfin.cpp.Mesh>`.

    *Example of usage*

        .. code-block:: python

            mesh = UnitSquare(4,4)
            vol = MinFacetEdgeLength(mesh)

    """

    return ufl.MinFacetEdgeLength(_mesh2domain(mesh))


# Simple definition of MaxFacetEdgeLength via UFL
def MaxFacetEdgeLength(mesh):
    """Return symbolic maximum facet edge length of a cell
    for given mesh.

    *Arguments*
        mesh
            a :py:class:`Mesh <dolfin.cpp.Mesh>`.

    *Example of usage*

        .. code-block:: python

            mesh = UnitSquare(4,4)
            maxfe = MaxFacetEdgeLength(mesh)

    """

    return ufl.MaxFacetEdgeLength(_mesh2domain(mesh))
