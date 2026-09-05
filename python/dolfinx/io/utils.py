# Copyright (C) 2017-2022 Chris N. Richardson, Garth N. Wells,
# Michal Habera and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""IO module for input data and post-processing file output."""

from pathlib import Path
from types import TracebackType
from typing import Self

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
from dolfinx.mesh import CellType, Geometry, GhostMode, Mesh, MeshTags

__all__ = ["VTKFile", "XDMFFile", "cell_perm_gmsh", "cell_perm_vtk", "distribute_entity_data"]


# VTXWriter requires ADIOS2
if _cpp.common.has_adios2:
    from dolfinx.cpp.io import VTXMeshPolicy

    __all__ = [*__all__, "VTXWriter", "VTXMeshPolicy"]

    class VTXWriter:
        """Writer for VTX files, using ADIOS2 to create the files.

        VTX supports arbitrary order Lagrange finite elements for the
        geometry description and arbitrary order (discontinuous)
        Lagrange finite elements for Functions.

        The files can be viewed using Paraview.
        """

        _cpp_object: _cpp.io.VTXWriter_float32 | _cpp.io.VTXWriter_float64

        def __init__(
            self,
            comm: _MPI.Comm,
            filename: str | Path,
            output: Mesh | Function | list[Function] | tuple[Function],
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
            if isinstance(output, Mesh):
                dtype = output.geometry.x.dtype
            elif isinstance(output, Function):
                dtype = output.function_space.mesh.geometry.x.dtype
            else:
                dtype = output[0].function_space.mesh.geometry.x.dtype

            if np.issubdtype(dtype, np.float32):
                _vtxwriter = _cpp.io.VTXWriter_float32
            elif np.issubdtype(dtype, np.float64):
                _vtxwriter = _cpp.io.VTXWriter_float64  # type: ignore[assignment]
            else:
                raise RuntimeError(f"VTXWriter does not support dtype={dtype}.")

            if isinstance(output, Mesh):
                self._cpp_object = _vtxwriter(comm, filename, output._cpp_object, engine)  # type: ignore[arg-type]
            else:
                cpp_objects = (
                    [output._cpp_object]
                    if isinstance(output, Function)
                    else [o._cpp_object for o in output]
                )
                self._cpp_object = _vtxwriter(comm, filename, cpp_objects, engine, mesh_policy)  # type: ignore[arg-type]

        def __enter__(self) -> Self:
            """Enter context manager."""
            return self

        def __exit__(
            self,
            exception_type: type[BaseException] | None,
            exception_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            """Exit context manager and close file."""
            self.close()

        def write(self, t: float) -> None:
            """Write data to file for a given time."""
            self._cpp_object.write(t)

        def close(self) -> None:
            """Close the VTX file."""
            self._cpp_object.close()


class VTKFile:
    """Interface to VTK files.

    VTK supports arbitrary order Lagrange finite elements for the
    geometry description. XDMF is the preferred format for geometry
    order <= 2.
    """

    _cpp_object: _cpp.io.VTKFile

    def __init__(self, comm: _MPI.Comm, filename: str | Path, mode: str):
        """Open a VTK file.

        Args:
            comm: MPI communicator used when opening the file.
            filename: Name of the file to open.
            mode: File opening mode, e.g. ``"w"`` for writing.
        """
        self._cpp_object = _cpp.io.VTKFile(comm, filename, mode)

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit context manager and close file."""
        self.close()

    def close(self) -> None:
        """Close the VTK file."""
        self._cpp_object.close()

    def write_mesh(self, mesh: Mesh, t: float = 0.0) -> None:
        """Write mesh to file for a given time."""
        self._cpp_object.write(mesh._cpp_object, t)

    def write_function(self, u: list[Function] | Function, t: float = 0.0) -> None:
        """Write a functions to file with a given time."""
        cpp_objects = [u._cpp_object] if isinstance(u, Function) else [_u._cpp_object for _u in u]
        self._cpp_object.write(cpp_objects, t)  # type: ignore[arg-type]


class XDMFFile:
    """Interface to manage XDMF files."""

    Encoding = _cpp.io.XDMFFile.Encoding

    _cpp_object: _cpp.io.XDMFFile

    def __init__(
        self,
        comm: _MPI.Comm,
        filename: str | Path,
        file_mode: str,
        encoding: _cpp.io.XDMFFile.Encoding = Encoding.HDF5,
    ):
        """Open an XDMF file.

        Args:
            comm: MPI communicator used when opening the file.
            filename: Name of the file to open.
            file_mode: File opening mode, e.g. ``"r"`` for reading or
                ``"w"`` for writing.
            encoding: File encoding.
        """
        self._cpp_object = _cpp.io.XDMFFile(comm, filename, file_mode, encoding)

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit context manager and close file."""
        self.close()

    def close(self) -> None:
        """Close the XDMF file."""
        self._cpp_object.close()

    @property
    def comm(self) -> _MPI.Comm:
        """MPI communicator that the file was opened with."""
        return self._cpp_object.comm

    def flush(self) -> None:
        """Flush any buffered data to file."""
        self._cpp_object.flush()

    def write_information(self, name: str, value: str, xpath: str = "/Xdmf/Domain") -> None:
        """Write information key-value pair to file.

        Args:
            name: Name of the key.
            value: Value of the key.
            xpath: XPath where the information is stored in the file.
        """
        self._cpp_object.write_information(name, value, xpath)

    def read_information(self, name: str, xpath: str = "/Xdmf/Domain") -> str:
        """Read information from file.

        Args:
            name: Name of the key to read.
            xpath: XPath where the information is stored in the file.

        Returns:
            Value associated with ``name``.
        """
        return self._cpp_object.read_information(name, xpath)

    def write_geometry(
        self, geometry: Geometry, name: str = "geometry", xpath: str = "/Xdmf/Domain"
    ) -> None:
        """Write mesh geometry to file.

        Args:
            geometry: Mesh geometry to write.
            name: Name of the grid node in the xml-scheme in the file.
            xpath: XPath where the Geometry Grid is stored in the file.
        """
        cpp_geometry = geometry._cpp_object
        if not isinstance(cpp_geometry, _cpp.mesh.Geometry_float64):
            raise TypeError("write_geometry currently only supports float64 geometry.")
        self._cpp_object.write_geometry(cpp_geometry, name, xpath)

    def read_topology_data(
        self, name: str = "mesh", xpath: str = "/Xdmf/Domain"
    ) -> npt.NDArray[np.int64]:
        """Read topology (cell connectivity) data for a mesh from file.

        Args:
            name: Name of the grid node in the xml-scheme in the file.
            xpath: XPath where the Mesh Grid is stored in the file.

        Returns:
            Cell connectivity data.
        """
        return self._cpp_object.read_topology_data(name, xpath)

    def read_geometry_data(
        self, name: str = "mesh", xpath: str = "/Xdmf/Domain"
    ) -> npt.NDArray[np.float64]:
        """Read geometry (node coordinates) data for a mesh from file.

        Args:
            name: Name of the grid node in the xml-scheme in the file.
            xpath: XPath where the Mesh Grid is stored in the file.

        Returns:
            Node coordinates.
        """
        return self._cpp_object.read_geometry_data(name, xpath)

    def read_cell_type(
        self, name: str = "mesh", xpath: str = "/Xdmf/Domain"
    ) -> tuple[CellType, int]:
        """Read the cell type and polynomial degree for a mesh from file.

        Args:
            name: Name of the grid node in the xml-scheme in the file.
            xpath: XPath where the Mesh Grid is stored in the file.

        Returns:
            Cell type and polynomial degree.
        """
        return self._cpp_object.read_cell_type(name, xpath)

    def write_mesh(self, mesh: Mesh, xpath: str = "/Xdmf/Domain") -> None:
        """Write mesh to file."""
        self._cpp_object.write_mesh(mesh._cpp_object, xpath)

    def write_meshtags(
        self,
        tags: MeshTags,
        x: Geometry,
        geometry_xpath: str = "/Xdmf/Domain/Grid/Geometry",
        xpath: str = "/Xdmf/Domain",
    ) -> None:
        """Write mesh tags to file."""
        if not isinstance(tags._cpp_object, _cpp.mesh.MeshTags_int32):
            raise TypeError("XDMF meshtags can only be written for int32-valued MeshTags.")
        self._cpp_object.write_meshtags(tags._cpp_object, x._cpp_object, geometry_xpath, xpath)

    def write_function(
        self,
        u: Function,
        t: float = 0.0,
        mesh_xpath: str = "/Xdmf/Domain/Grid[@GridType='Uniform'][1]",
    ) -> None:
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
        self._cpp_object.write_function(u._cpp_object, t, mesh_xpath)

    def read_mesh(
        self,
        ghost_mode: GhostMode = GhostMode.shared_facet,
        name: str = "mesh",
        xpath: str = "/Xdmf/Domain",
        max_facet_to_cell_links: int = 2,
    ) -> Mesh:
        """Read mesh data from file.

        Note:
            Changing `max_facet_to_cell_links` from the default value
            should only be required when working on branching manifolds.
            Changing this value on non-branching meshes will only result in
            a slower mesh partitioning and creation.

        Args:
            ghost_mode: Ghost mode to use for the cells in mesh creation.
            name: Name of the grid node in the xml-scheme in the file
            xpath: XPath where Mesh Grid is stored in the file.
            max_facet_to_cell_links: Maximum number of cells that a facet
                can be linked to.
        """
        cell_shape, cell_degree = self.read_cell_type(name, xpath)
        cells = self.read_topology_data(name, xpath)
        x = self.read_geometry_data(name, xpath)

        # Get coordinate element, special handling for second order
        # serendipity.
        basix_el: basix.ufl._BasixElement | basix.ufl._BlockedElement
        num_nodes_per_cell = cells.shape[1]
        if (cell_shape == CellType.quadrilateral and num_nodes_per_cell == 8) or (
            cell_shape == CellType.hexahedron and num_nodes_per_cell == 20
        ):
            s_el = basix.ufl.element(
                basix.ElementFamily.serendipity,
                cell_shape.name,
                2,
            )
            # Create a custom element that is serendipity but uses points
            # evaluations on edges
            geometry = basix.cell.geometry(s_el.basix_element.cell_type)
            topology = basix.cell.topology(s_el.basix_element.cell_type)
            e_x: list[list[npt.NDArray[np.floating]]] = [
                [np.array([p]) for p in geometry],
                [np.array([(geometry[edge[0]] + geometry[edge[1]]) / 2]) for edge in topology[1]],
                [np.zeros((0, 3)) for _ in s_el.basix_element.x[2]],
                [np.zeros((0, 3)) for _ in s_el.basix_element.x[3]],
            ]
            e_m: list[list[npt.NDArray[np.floating]]] = [
                [np.ones((1, 1, 1, 1)) for _ in s_el.basix_element.M[0]],
                [np.ones((1, 1, 1, 1)) for _ in s_el.basix_element.M[1]],
                [np.zeros((0, 1, 0, 1)) for _ in s_el.basix_element.M[2]],
                [np.zeros((0, 1, 0, 1)) for _ in s_el.basix_element.M[3]],
            ]
            el = basix.ufl.custom_element(
                s_el.basix_element.cell_type,
                s_el.reference_value_shape,
                s_el.basix_element.wcoeffs,
                e_x,
                e_m,
                0,
                s_el.map_type,
                s_el.basix_element.sobolev_space,
                s_el.discontinuous,
                s_el.embedded_subdegree,
                s_el.embedded_superdegree,
                s_el.polyset_type,
                s_el.dtype,
            )
            cmap = _cpp.fem.CoordinateElement_float64(el.basix_element._e)
            basix_el = basix.ufl.blocked_element(el, shape=(x.shape[1],))
        else:
            basix_el = basix.ufl.element(
                "Lagrange",
                cell_shape.name,
                cell_degree,
                basix.LagrangeVariant.unset,
                shape=(x.shape[1],),
            )
            cmap = _cpp.fem.CoordinateElement_float64(cell_shape, cell_degree)

        # Build the mesh
        msh = _cpp.mesh.create_mesh(
            self.comm,
            cells,
            cmap,
            x,
            _cpp.graph.partitioner(),
            ghost_mode,
            max_facet_to_cell_links,
            1,
            cell_weights=None,
            reorder_fn=None,
        )
        msh.name = name
        domain = ufl.Mesh(basix_el)
        return Mesh(msh, domain)

    def read_meshtags(
        self,
        mesh: Mesh,
        name: str,
        attribute_name: str | None = None,
        xpath: str = "/Xdmf/Domain",
    ) -> MeshTags:
        """Read mesh tags with name given in the XMDF file.

        Args:
            mesh: Mesh that the input data is defined on.
            name: Name of the grid node in the xml-scheme of the
                XDMF-file.
            attribute_name: The name of the attribute to read. If
                ``attribute_name`` is empty, reads the first attribute in
                the file. If ``attribute_name`` is not empty but no
                attributes have the provided name, throws an error. If
                multiple attributes have the provided name, reads the
                first one found.
            xpath: XPath where MeshTags Grid is stored in file.

        Returns:
            A MeshTags object containing the requested data read from
            file.
        """
        cpp_mesh = mesh._cpp_object
        if not isinstance(cpp_mesh, _cpp.mesh.Mesh_float64):
            raise TypeError("read_meshtags currently only supports float64 meshes.")
        mt = self._cpp_object.read_meshtags(cpp_mesh, name, attribute_name, xpath)
        return MeshTags(mt)


def distribute_entity_data(
    mesh: Mesh, entity_dim: int, entities: npt.NDArray[np.int64], values: np.ndarray
) -> tuple[npt.NDArray[np.int64], np.ndarray]:
    """Distribute  mesh entities and values to owning process.

    The entities are described by the global vertex indices of the mesh.
    These entity indices are using the original input ordering.

    Returns:
        Entities owned by the process (and their local entity-to-vertex
        indices) and the corresponding values.
    """
    return _cpp.io.distribute_entity_data(
        mesh.topology._cpp_object,
        mesh.geometry.input_global_indices,
        mesh.geometry.index_map().size_global,
        mesh.geometry.cmaps[0].create_dof_layout(),
        mesh.geometry.dofmaps[0],
        entity_dim,
        entities,
        values,
    )
