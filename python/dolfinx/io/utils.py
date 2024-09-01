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
from dolfinx.mesh import GhostMode, Mesh, MeshTags

__all__ = [
    "VTKFile",
    "XDMFFile",
    "cell_perm_gmsh",
    "cell_perm_vtk",
    "distribute_entity_data",
]


def _extract_cpp_objects(functions: typing.Union[Mesh, Function, tuple[Function], list[Function]]):
    """Extract C++ objects"""
    if isinstance(functions, (list, tuple)):
        return [getattr(u, "_cpp_object", u) for u in functions]
    else:
        return [getattr(functions, "_cpp_object", functions)]


# FidesWriter and VTXWriter require ADIOS2
if _cpp.common.has_adios2:
    from dolfinx.cpp.io import FidesMeshPolicy, VTXMeshPolicy  # F401

    __all__ = [
        *__all__,
        "FidesWriter",
        "VTXWriter",
        "FidesMeshPolicy",
        "VTXMeshPolicy",
        "ADIOS2",
        "read_mesh",
        "read_meshtags",
        "read_timestamps",
        "update_mesh",
        "write_mesh",
        "write_meshtags",
    ]

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

    class FidesWriter:
        """Writer for Fides files, using ADIOS2 to create the files.

        Fides (https://fides.readthedocs.io/) supports first order
        Lagrange finite elements for the geometry description and first
        order Lagrange finite elements for functions. All functions have
        to be of the same element family and same order.

        The files can be displayed by Paraview.
        """

        _cpp_object: typing.Union[_cpp.io.FidesWriter_float32, _cpp.io.FidesWriter_float64]

        def __init__(
            self,
            comm: _MPI.Comm,
            filename: typing.Union[str, Path],
            output: typing.Union[Mesh, list[Function], Function],
            engine: str = "BPFile",
            mesh_policy: FidesMeshPolicy = FidesMeshPolicy.update,
        ):
            """Initialize a writer for outputting a mesh, a single Lagrange
            function or list of Lagrange functions sharing the same
            element family and degree

            Args:
                comm: MPI communicator.
                filename: Output filename.
                output: Data to output. Either a mesh, a single degree one
                    Lagrange function or list of degree one Lagrange functions.
                engine: ADIOS2 engine to use for output. See
                    ADIOS2 documentation for options.
                mesh_policy: Controls if the mesh is written to file at
                    the first time step only when a ``Function`` is
                    written to file, or is re-written (updated) at each
                    time step. Has an effect only for ``Function``
                    output.
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
                _fides_writer = _cpp.io.FidesWriter_float32
            elif np.issubdtype(dtype, np.float64):
                _fides_writer = _cpp.io.FidesWriter_float64

            try:
                self._cpp_object = _fides_writer(comm, filename, output._cpp_object, engine)  # type: ignore
            except (NotImplementedError, TypeError, AttributeError):
                self._cpp_object = _fides_writer(
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

    class ADIOS2(_cpp.io.ADIOS2):
        """Wrapper around ADIOS2.
        Supports creation of arbitrary number of IOs
        """

        def __enter__(self):
            return self

        def __exit__(self, exception_type, exception_value, traceback):
            self.close()

        def add_io(self, filename: str, tag: str, engine_type: str = "BP5", mode: str = "append"):
            """Add IO and Engine

            Args:
                filename: filename to associate Engine with
                tag: Name of the IO to use
                engine_type: ADIOS2 Engine type, default is "BP5"
                mode: ADIOS2 mode, default is "append"
            """
            super().add_io(filename, tag, engine_type, mode)

        def close(self, tag: str):
            """Close IO and Engine associated with the given tag"""
            super().close(tag)

    def write_mesh(ADIOS2: ADIOS2, tag: str, mesh: Mesh, time: np.floating = 0) -> None:
        """Write mesh to a file using ADIOS2

        Args:
            ADIOS2: Wrapper around ADIOS2.
            tag: Name of the IO to use.
            mesh: Mesh to write to the file.
            time: Associated time stamp
        """

        dtype = mesh.geometry.x.dtype  # type: ignore
        if np.issubdtype(dtype, np.float32):
            _writer = _cpp.io.write_mesh_float32
        elif np.issubdtype(dtype, np.float64):
            _writer = _cpp.io.write_mesh_float64

        return _writer(ADIOS2, tag, mesh._cpp_object, time)

    def read_mesh(
        ADIOS2: ADIOS2, tag: str, comm: _MPI.Comm, ghost_mode: GhostMode = GhostMode.shared_facet
    ) -> Mesh:
        """Read mesh from a file using ADIOS2

        Args:
            ADIOS2: Wrapper around ADIOS2
            tag: Name of the IO to use
            comm: The MPI communicator
            ghost_mode: GhostMode to use on mesh partitioning
        Returns:
            Mesh
        """
        msh = _cpp.io.read_mesh(ADIOS2, tag, comm, ghost_mode)

        element = basix.ufl.element(
            basix.ElementFamily.P,
            basix.CellType[msh.topology.cell_name()],
            msh.geometry.cmap.degree,
            basix.LagrangeVariant(int(msh.geometry.cmap.variant)),
            shape=(msh.geometry.x.shape[1],),
            dtype=msh.geometry.x.dtype,
        )
        domain = ufl.Mesh(element)

        return Mesh(msh, domain)

    def write_meshtags(ADIOS2: ADIOS2, tag: str, mesh: Mesh, meshtags: MeshTags) -> None:
        """
        Write meshtags to a file using ADIOS2

        Args:
            ADIOS2: Wrapper to ADIOS2
            tag: Name of the IO to use
            mesh: The mesh
            meshtags: The associated meshtags
        """

        return _cpp.io.write_meshtags(ADIOS2, tag, mesh._cpp_object, meshtags._cpp_object)

    def read_meshtags(ADIOS2: ADIOS2, tag: str, mesh: Mesh) -> dict[str, MeshTags]:
        """
        Read all meshtags from a file using ADIOS2

        Args:
            ADIOS2: Wrapper to ADIOS2
            tag: Name of the IO to use
            mesh: The mesh
        Returns:
            meshtags: All meshtags stored in the file are returned as a dictionary
            with (meshtags_name, meshtags) key-value pairs.
        """

        tags = _cpp.io.read_meshtags(ADIOS2, tag, mesh._cpp_object)

        for name, mt_cpp in tags.items():
            tags[name] = MeshTags(mt_cpp)

        return tags

    def read_timestamps(ADIOS2: ADIOS2, tag: str) -> np.ndarray:
        """Read timestamps from a file using ADIOS2

        Args:
            ADIOS2: Wrapper around ADIOS2
            tag: Name of the IO to use
        Returns:
            vector of timestamps
        """
        return _cpp.io.read_timestamps(ADIOS2, tag)

    def update_mesh(ADIOS2: ADIOS2, tag: str, mesh: Mesh, step: np.int64) -> None:
        """Update mesh geometry with geometry stored with the given time stamp in the file

        Args:
            ADIOS2: Wrapper around ADIOS2
            tag: Name of the IO to use
            mesh: Mesh to update the geometry
            step: ADIOS2 Step at which the corresponding geometry is to be read from the file
        """

        return _cpp.io.update_mesh(ADIOS2, tag, mesh._cpp_object, step)


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
        x: typing.Union[_cpp.mesh.Geometry_float32, _cpp.mesh.Geometry_float64],
        geometry_xpath: str = "/Xdmf/Domain/Grid/Geometry",
        xpath: str = "/Xdmf/Domain",
    ) -> None:
        """Write mesh tags to file"""
        super().write_meshtags(tags._cpp_object, x, geometry_xpath, xpath)

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

    def read_meshtags(self, mesh, name, xpath="/Xdmf/Domain"):
        mt = super().read_meshtags(mesh._cpp_object, name, xpath)
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
        mesh.topology,
        mesh.geometry.input_global_indices,
        mesh.geometry.index_map().size_global,
        mesh.geometry.cmap.create_dof_layout(),
        mesh.geometry.dofmap,
        entity_dim,
        entities,
        values,
    )
