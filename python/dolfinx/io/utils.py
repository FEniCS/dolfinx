# Copyright (C) 2017-2022 Chris N. Richardson, Garth N. Wells, Michal Habera
# and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""IO module for input data and post-processing file output"""

import typing

import basix
import basix.ufl_wrapper
import numpy as np
import numpy.typing as npt
import ufl
from dolfinx.cpp.io import perm_gmsh as cell_perm_gmsh  # noqa F401
from dolfinx.cpp.io import perm_vtk as cell_perm_vtk  # noqa F401
from dolfinx.fem import Function
from dolfinx.mesh import GhostMode, Mesh, MeshTags
from mpi4py import MPI as _MPI

from dolfinx import cpp as _cpp

__all__ = ["VTKFile", "XDMFFile", "cell_perm_gmsh", "cell_perm_vtk",
           "distribute_entity_data"]


def _extract_cpp_functions(functions: typing.Union[typing.List[Function], Function]):
    """Extract C++ object for a single function or a list of functions"""
    if isinstance(functions, (list, tuple)):
        return [getattr(u, "_cpp_object", u) for u in functions]
    else:
        return [getattr(functions, "_cpp_object", functions)]


if _cpp.common.has_adios2:
    # FidesWriter and VTXWriter require ADIOS2
    __all__ = __all__ + ["FidesWriter", "VTXWriter"]

    class VTXWriter(_cpp.io.VTXWriter):
        """Interface to VTK files for ADIOS2

        VTX supports arbitrary order Lagrange finite elements for the
        geometry description and arbitrary order (discontinuous)
        Lagrange finite elements for Functions.

        The files can be displayed by Paraview. The storage backend uses
        ADIOS2.

        """

        def __init__(self, comm: _MPI.Comm, filename: str, output: typing.Union[Mesh, Function, typing.List[Function]]):
            """Initialize a writer for outputting data in the VTX format.

            Args:
                comm: The MPI communicator
                filename: The output filename
                output: The data to output. Either a mesh, a single
                    (discontinuous) Lagrange Function or list of
                    (discontinuous Lagrange Functions.

            Note:
                All Functions for output must share the same mesh and
                have the same element type.

            """
            try:
                # Input is a mesh
                super().__init__(comm, filename, output._cpp_object)  # type: ignore[union-attr]
            except (NotImplementedError, TypeError, AttributeError):
                # Input is a single function or a list of functions
                super().__init__(comm, filename, _extract_cpp_functions(output))   # type: ignore[arg-type]

        def __enter__(self):
            return self

        def __exit__(self, exception_type, exception_value, traceback):
            self.close()

    class FidesWriter(_cpp.io.FidesWriter):
        """Interface to Fides file format.

        Fides supports first order Lagrange finite elements for the
        geometry descriptionand first order Lagrange finite elements for
        functions. All functions has to be of the same element family
        and same order.

        The files can be displayed by Paraview. The storage backend uses
        ADIOS2.

        """

        def __init__(self, comm: _MPI.Comm, filename: str, output: typing.Union[Mesh, typing.List[Function], Function]):
            """Initialize a writer for outputting a mesh, a single Lagrange
            function or list of Lagrange functions sharing the same
            element family and degree

            Args:
                comm: The MPI communicator
                filename: The output filename
                output: The data to output. Either a mesh, a single
                    first order Lagrange function or list of first order
                    Lagrange functions.

            """
            try:
                super().__init__(comm, filename, output._cpp_object)  # type: ignore[union-attr]
            except (NotImplementedError, TypeError, AttributeError):
                super().__init__(comm, filename, _extract_cpp_functions(output))  # type: ignore[arg-type]

        def __enter__(self):
            return self

        def __exit__(self, exception_type, exception_value, traceback):
            self.close()


class VTKFile(_cpp.io.VTKFile):
    """Interface to VTK files

    VTK supports arbitrary order Lagrange finite elements for the
    geometry description. XDMF is the preferred format for geometry
    order <= 2.

    """

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def write_mesh(self, mesh: Mesh, t: float = 0.0) -> None:
        """Write mesh to file for a given time (default 0.0)"""
        self.write(mesh._cpp_object, t)

    def write_function(self, u: typing.Union[typing.List[Function], Function], t: float = 0.0) -> None:
        """Write a single function or a list of functions to file for a given time (default 0.0)"""
        super().write(_extract_cpp_functions(u), t)


class XDMFFile(_cpp.io.XDMFFile):
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def write_mesh(self, mesh: Mesh, xpath: str = "/Xdmf/Domain") -> None:
        """Write mesh to file for a given time (default 0.0)"""
        super().write_mesh(mesh._cpp_object, xpath)

    def write_meshtags(self, tags: MeshTags, geometry_xpath: str = "/Xdmf/Domain/Grid/Geometry",
                       xpath: str = "/Xdmf/Domain") -> None:
        """Write mesh tags to file for a given time (default 0.0)"""
        super().write_meshtags(tags._cpp_object, geometry_xpath, xpath)

    def write_function(self, u, t: float = 0.0, mesh_xpath="/Xdmf/Domain/Grid[@GridType='Uniform'][1]"):
        super().write_function(getattr(u, "_cpp_object", u), t, mesh_xpath)

    def read_mesh(self, ghost_mode=GhostMode.shared_facet, name="mesh", xpath="/Xdmf/Domain") -> Mesh:
        """Read mesh data from file"""
        cell_shape, cell_degree = super().read_cell_type(name, xpath)
        cells = super().read_topology_data(name, xpath)
        x = super().read_geometry_data(name, xpath)

        # Build the mesh
        cmap = _cpp.fem.CoordinateElement(cell_shape, cell_degree)
        msh = _cpp.mesh.create_mesh(self.comm(), _cpp.graph.AdjacencyList_int64(cells),
                                    cmap, x, _cpp.mesh.create_cell_partitioner(ghost_mode))
        msh.name = name

        domain = ufl.Mesh(basix.ufl_wrapper.create_vector_element(
            "Lagrange", cell_shape.name, cell_degree, basix.LagrangeVariant.equispaced, dim=x.shape[1],
            gdim=x.shape[1]))
        return Mesh(msh, domain)

    def read_meshtags(self, mesh, name, xpath="/Xdmf/Domain"):
        mt = super().read_meshtags(mesh._cpp_object, name, xpath)
        return MeshTags(mt, mesh)


def distribute_entity_data(mesh: Mesh, entity_dim: int, entities: npt.NDArray[np.int64],
                           values: npt.NDArray[np.int32]) -> typing.Tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]]:
    return _cpp.io.distribute_entity_data(mesh._cpp_object, entity_dim, entities, values)
