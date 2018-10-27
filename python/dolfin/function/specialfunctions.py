# -*- coding: utf-8 -*-
# Copyright (C) 2008-2014 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Some special functions"""

import ufl
from dolfin import cpp, function


class MeshCoordinates(function.BaseExpression):
    def __init__(self, mesh: cpp.mesh.Mesh):
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


class FacetArea(function.BaseExpression):
    def __init__(self, mesh: cpp.mesh.Mesh):
        """Create function that evaluates to the facet area/length on each
        facet.

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
        ufl_element = ufl.FiniteElement("Discontinuous Lagrange",
                                        mesh.ufl_cell(), 0)
        super().__init__(
            domain=mesh.ufl_domain(), element=ufl_element, name="FacetArea")


def FacetNormal(mesh: cpp.mesh.Mesh) -> ufl.FacetNormal:
    """Return symbolic facet normal for given mesh.

    *Example of usage*

        .. code-block:: python

            mesh = UnitSquare(4,4)
            n = FacetNormal(mesh)

    """

    return ufl.FacetNormal(mesh.ufl_domain())


def CellDiameter(mesh: cpp.mesh.Mesh) -> ufl.CellDiameter:
    r"""Return function cell diameter for given mesh.

    Note that diameter of cell :math:`K` is defined as
    :math:`\sup_{\mathbf{x, y} \in K} |\mathbf{x - y}|`.

    *Example of usage*

        .. code-block:: python

            mesh = UnitSquare(4,4)
            h = CellDiameter(mesh)

    """

    return ufl.CellDiameter(mesh.ufl_domain())


def CellVolume(mesh: cpp.mesh.Mesh) -> ufl.CellVolume:
    """Return symbolic cell volume for given mesh.

    *Example of usage*

        .. code-block:: python

            mesh = UnitSquare(4,4)
            vol = CellVolume(mesh)

    """

    return ufl.CellVolume(mesh.ufl_domain())


def SpatialCoordinate(mesh: cpp.mesh.Mesh) -> ufl.SpatialCoordinate:
    """Return symbolic physical coordinates for given mesh.

    *Example of usage*

        .. code-block:: python

            mesh = UnitSquare(4,4)
            x = SpatialCoordinate(mesh)

    """

    return ufl.SpatialCoordinate(mesh.ufl_domain())


def CellNormal(mesh: cpp.mesh.Mesh) -> ufl.CellNormal:
    """Return symbolic cell normal for given manifold mesh.

    *Example of usage*

        .. code-block:: python

            mesh = UnitSquare(4,4)
            n = CellNormal(mesh)

    """

    return ufl.CellNormal(mesh.ufl_domain())


def Circumradius(mesh: cpp.mesh.Mesh) -> ufl.Circumradius:
    """Return symbolic cell circumradius for given mesh.

    *Example of usage*

        .. code-block:: python

            mesh = UnitSquare(4,4)
            R = Circumradius(mesh)

    """

    return ufl.Circumradius(mesh.ufl_domain())


def MinCellEdgeLength(mesh: cpp.mesh.Mesh) -> ufl.MinCellEdgeLength:
    """Return symbolic minimum cell edge length of a cell
    for given mesh.

    *Example of usage*

        .. code-block:: python

            mesh = UnitSquare(4,4)
            mince = MinCellEdgeLength(mesh)

    """

    return ufl.MinCellEdgeLength(mesh.ufl_domain())


def MaxCellEdgeLength(mesh: cpp.mesh.Mesh) -> ufl.MaxCellEdgeLength:
    """Return symbolic maximum cell edge length of a cell
    for given mesh.

    *Example of usage*

        .. code-block:: python

            mesh = UnitSquare(4,4)
            maxce = MaxCellEdgeLength(mesh)

    """

    return ufl.MaxCellEdgeLength(mesh.ufl_domain())


def MinFacetEdgeLength(mesh: cpp.mesh.Mesh) -> ufl.MinFacetEdgeLength:
    """Return symbolic minimum facet edge length of a cell
    for given mesh.

    *Example of usage*

        .. code-block:: python

            mesh = UnitSquare(4,4)
            vol = MinFacetEdgeLength(mesh)

    """

    return ufl.MinFacetEdgeLength(mesh.ufl_domain())


def MaxFacetEdgeLength(mesh: cpp.mesh.Mesh) -> ufl.MaxFacetEdgeLength:
    """Return symbolic maximum facet edge length of a cell
    for given mesh.

    *Example of usage*

        .. code-block:: python

            mesh = UnitSquare(4,4)
            maxfe = MaxFacetEdgeLength(mesh)

    """

    return ufl.MaxFacetEdgeLength(mesh.ufl_domain())
