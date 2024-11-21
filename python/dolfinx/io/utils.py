# Copyright (C) 2017-2022 Chris N. Richardson, Garth N. Wells, Michal Habera
# and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""IO module for input data and post-processing file output."""

import typing
from pathlib import Path

from mpi4py import MPI as _MPI

import numpy as np
import numpy.typing as npt

import basix
import basix.ufl
import ufl
from dolfinx import cpp as _cpp
from dolfinx.cpp.io import perm_gmsh as cell_perm_gmsh
from dolfinx.cpp.io import perm_vtk as cell_perm_vtk
from dolfinx.fem import Function
from dolfinx.mesh import Geometry, GhostMode, Mesh, MeshTags

__all__ = ["VTKFile", "XDMFFile", "cell_perm_gmsh", "cell_perm_vtk", "distribute_entity_data"]


def _extract_cpp_objects(functions: typing.Union[Mesh, Function, tuple[Function], list[Function]]):
    """Extract C++ objects"""
    if isinstance(functions, (list, tuple)):
        return [getattr(u, "_cpp_object", u) for u in functions]
    else:
        return [getattr(functions, "_cpp_object", functions)]


# VTXWriter requires ADIOS2
if _cpp.common.has_adios2:
    from dolfinx.cpp.io import VTXMeshPolicy  # F401

    __all__ = [*__all__, "VTXWriter", "VTXMeshPolicy"]

    class VTXWriter:
        """Writer for VTX files, using ADIOS2 to create the files.

        VTX supports arbitrary order Lagrange finite elements for the
        geometry description and arbitrary order (discontinuous)
        Lagrange finite elements for Functions.

        The files can be viewed using Paraview.
        """

        _cpp_object: typing.Union[_cpp.io.VTXWriter_float32, _cpp.io.VTXWriter_float64]

        def __init__(
            self,
            comm: _MPI.Comm,
            filename: typing.Union[str, Path],
            output: typing.Union[Mesh, Function, list[Function], tuple[Function]],
            engine: str = "BPFile",
            mesh_policy: VTXMeshPolicy = VTXMeshPolicy.update,
        ):
            """Initialize a writer for outputting data in the VTX format.

            Args:
                comm: The MPI communicator
                filename: The output filename
                output: The data to output. Either a mesh, a single
                    (discontinuous) Lagrange Function or list of
                    (discontinuous) Lagrange Functions.
                engine: ADIOS2 engine to use for output. See
                    ADIOS2 documentation for options.
                mesh_policy: Controls if the mesh is written to file at
                    the first time step only when a ``Function`` is
                    written to file, or is re-written (updated) at each
                    time step. Has an effect only for ``Function``
                    output.

            Note:
                All Functions for output must share the same mesh and
                have the same element type.
            """
            # Get geometry type
            try:
                dtype = output.geometry.x.dtype  # type: ignore
            except AttributeError:
                try:
                    dtype = output.function_space.mesh.geometry.x.dtype  # type: ignore
                except AttributeError:
                    dtype = output[0].function_space.mesh.geometry.x.dtype  # type: ignore

            if np.issubdtype(dtype, np.float32):
                _vtxwriter = _cpp.io.VTXWriter_float32
            elif np.issubdtype(dtype, np.float64):
                _vtxwriter = _cpp.io.VTXWriter_float64

            try:
                # Input is a mesh
                self._cpp_object = _vtxwriter(comm, filename, output._cpp_object, engine)  # type: ignore[union-attr]
            except (NotImplementedError, TypeError, AttributeError):
                # Input is a single function or a list of functions
                self._cpp_object = _vtxwriter(
                    comm, filename, _extract_cpp_objects(output), engine, mesh_policy
                )  # type: ignore[arg-type]

        def __enter__(self):
            return self

        def __exit__(self, exception_type, exception_value, traceback):
            self.close()

        def write(self, t: float):
            self._cpp_object.write(t)

        def close(self):
            self._cpp_object.close()


class VTKFile(_cpp.io.VTKFile):
    """Interface to VTK files.

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

    def write_function(self, u: typing.Union[list[Function], Function], t: float = 0.0) -> None:
        """Write a single function or a list of functions to file for a given time (default 0.0)"""
        super().write(_extract_cpp_objects(u), t)


class XDMFFile(_cpp.io.XDMFFile):
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def write_mesh(self, mesh: Mesh, xpath: str = "/Xdmf/Domain") -> None:
        """Write mesh to file"""
        super().write_mesh(mesh._cpp_object, xpath)

    def write_meshtags(
        self,
        tags: MeshTags,
        x: Geometry,
        geometry_xpath: str = "/Xdmf/Domain/Grid/Geometry",
        xpath: str = "/Xdmf/Domain",
    ) -> None:
        """Write mesh tags to file"""
        super().write_meshtags(tags._cpp_object, x._cpp_object, geometry_xpath, xpath)

    def write_function(
        self, u: Function, t: float = 0.0, mesh_xpath="/Xdmf/Domain/Grid[@GridType='Uniform'][1]"
    ):
        """Write function to file for a given time.

        Note:
            Function is interpolated onto the mesh nodes, as a Nth order
            Lagrange function, where N is the order of the coordinate
            map. If the Function is a cell-wise constant, it is saved as
            a cell-wise constant.

        Args:
            u: Function to write to file.
            t: Time associated with Function output.
            mesh_xpath: Path to mesh associated with the Function in the
                XDMFFile.
        """
        super().write_function(getattr(u, "_cpp_object", u), t, mesh_xpath)

    def read_mesh(
        self, ghost_mode=GhostMode.shared_facet, name="mesh", xpath="/Xdmf/Domain"
    ) -> Mesh:
        """Read mesh data from file."""
        cell_shape, cell_degree = super().read_cell_type(name, xpath)
        cells = super().read_topology_data(name, xpath)
        x = super().read_geometry_data(name, xpath)

        # Build the mesh
        cmap = _cpp.fem.CoordinateElement_float64(cell_shape, cell_degree)
        msh = _cpp.mesh.create_mesh(
            self.comm, cells, cmap, x, _cpp.mesh.create_cell_partitioner(ghost_mode)
        )
        msh.name = name
        domain = ufl.Mesh(
            basix.ufl.element(
                "Lagrange",
                cell_shape.name,
                cell_degree,
                basix.LagrangeVariant.equispaced,
                shape=(x.shape[1],),
            )
        )
        return Mesh(msh, domain)

    def read_meshtags(self, mesh, name, attribute_label="", xpath="/Xdmf/Domain"):
        """Read meshtags with a specific name as specified in the xdmf file. If
        `attribute_label` is empty, read the first attribute in the file.
        If `attribute_label` is not empty but no attributes have the provided name,
        throw an error. If multiple attributes have the provided name, read the first
        one found."""
        mt = super().read_meshtags(
            mesh._cpp_object, name, attribute_label, xpath
        )
        return MeshTags(mt)


def distribute_entity_data(
    mesh: Mesh, entity_dim: int, entities: npt.NDArray[np.int64], values: np.ndarray
) -> tuple[npt.NDArray[np.int64], np.ndarray]:
    """Given a set of mesh entities and values, distribute them to the process that owns the entity.

    The entities are described by the global vertex indices of the mesh. These entity indices are
    using the original input ordering.

    Returns:
        Entities owned by the process (and their local entity-to-vertex indices) and the
        corresponding values.
    """
    return _cpp.io.distribute_entity_data(
        mesh.topology._cpp_object,
        mesh.geometry.input_global_indices,
        mesh.geometry.index_map().size_global,
        mesh.geometry.cmap.create_dof_layout(),
        mesh.geometry.dofmap,
        entity_dim,
        entities,
        values,
    )
