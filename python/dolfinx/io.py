# Copyright (C) 2017-2018 Chris N. Richardson, Garth N. Wells and Michal Habera
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""IO module for input data, post-processing and checkpointing"""

import typing

import numpy

from dolfinx import cpp, fem, function

__all__ = ["HDF5File"]


class HDF5File:
    """Interface to HDF5 files"""

    def __init__(self, mpi_comm, filename: str, mode: str):
        """Open HDF5 file

        Parameters
        ----------
        mpi_comm
            The MPI communicator
        filename
            Name of the file
        mode
            File opening mode, which can be write (w), read (r) or append (a)

        """
        self._cpp_object = cpp.io.HDF5File(mpi_comm, filename, mode)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        return self._cpp_object.close()

    def get_mpi_atomicity(self) -> bool:
        """Get atomicity of the HDF5 file"""
        return self._cpp_object.get_mpi_atomicity()

    def set_mpi_atomicity(self, atomicity: bool) -> None:
        """Set atomicity of the HDF5 file"""
        self._cpp_object.set_mpi_atomicity(atomicity)

    def close(self) -> None:
        """Close file"""
        self._cpp_object.close()

    def write(self, o, name, t=None) -> None:
        """Write object to file"""
        o_cpp = getattr(o, "_cpp_object", o)
        if t is None:
            self._cpp_object.write(o_cpp, name)
        else:
            self._cpp_object.write(o_cpp, name, t)

    def read_mvc(self, mesh, name: str = ""):
        # FIXME: figure type out from file (or pass string)  and return?
        raise NotImplementedError("General MVC read function not implemented.")

    # def read_mvc(self, mesh, type: str, name: str = ""):
    #     # FIXME: return appropriate MVC based on type string
    #     raise NotImplementedError("General MVC read function not implemented.")

    # ----------------------------------------------------------

    # FIXME: implement a common function for multiple types

    def read_mvc_size_t(self, mesh, name: str = ""):
        """Read MeshValueCollection of type size_t"""
        return self._cpp_object.read_mvc_size_t(mesh, name)

    def read_mf_double(self, mesh, name: str = ""):
        """Read MeshFunction of type float"""
        return self._cpp_object.read_mf_double(mesh, name)

    # ----------------------------------------------------------

    def read_vector(self, mpi_comm, data_path: str,
                    use_partition_from_file: bool):
        """Read Vector"""
        return self._cpp_object.read_vector(mpi_comm, data_path,
                                            use_partition_from_file)

    def read_mesh(self, data_path: str, use_partition_from_file: bool,
                  ghost_mode):
        mesh = self._cpp_object.read_mesh(data_path, use_partition_from_file,
                                          ghost_mode)
        mesh.geometry.coord_mapping = fem.create_coordinate_map(mesh)
        return mesh

    def read_function(self, V, name: str):
        """Read finite element Function from file

        Parameters
        ----------
        V
            Function space of saved function.
        name
            Name of function as saved into HDF file.
        Returns
        -------
        dolfinx.function.function.Function
            Function read from file

        Note
        ----
        Parameter `V: Function space` must be the same as saved function space
        except for ordering of mesh entities.
        """

        V_cpp = getattr(V, "_cpp_object", V)
        u_cpp = self._cpp_object.read(V_cpp, name)
        return function.Function(V, u_cpp.vector)


class VTKFile:
    """Interface to VTK files
    VTK supports arbitrary order Lagrangian finite elements for the
    geometry description. XDMF is the preferred format for geometry
    order <= 2.

    """

    def __init__(self, filename: str):
        """Open VTK file
        Parameters
        ----------
        filename
            Name of the file
        """
        self._cpp_object = cpp.io.VTKFile(filename)

    def write(self, o, t=None) -> None:
        """Write object to file"""
        o_cpp = getattr(o, "_cpp_object", o)
        if t is None:
            self._cpp_object.write(o_cpp)
        else:
            self._cpp_object.write(o_cpp, t)


class XDMFFile:
    """Interface to XDMF files
    This format is preferred on lower order geometries and for
    DG and RT function spaces.
    XDMF also allows for checkpointing of solutions and has parallel support.

    """

    # Import encoding (find better way?)
    Encoding = cpp.io.XDMFFile.Encoding

    def __init__(self, mpi_comm, filename: str, encoding=Encoding.HDF5):
        """Open XDMF file

        Parameters
        ----------
        mpi_comm
            The MPI communicator
        filename
            Name of the file
        encoding
            Encoding used for 'heavy' data when writing/appending

        """
        self._cpp_object = cpp.io.XDMFFile(mpi_comm, filename, encoding)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        return self._cpp_object.close()

    def close(self) -> None:
        """Close file"""
        self._cpp_object.close()

    def write(self, o, t=None) -> None:
        """Write object to file

        Parameters
        ----------
        o
            The object to write to file
        t
            The time stamp

        """
        o_cpp = getattr(o, "_cpp_object", o)
        if t is None:
            self._cpp_object.write(o_cpp)
        else:
            self._cpp_object.write(o_cpp, t)

    def read_mesh(self):
        mesh = self._cpp_object.read_mesh()
        mesh.geometry.coord_mapping = fem.create_coordinate_map(mesh)
        return mesh

    def read_mesh_data(self) -> typing.Tuple[cpp.mesh.CellType, numpy.ndarray, numpy.ndarray]:
        """Read in mesh data

        Parameters
        ----------
        mpi_comm:
            MPI communicator
        Returns
        -------
        cell_type
            Cell type
        x
            Geometric points on each process
        cells
            Topological cells with global vertex indexing
        """
        return self._cpp_object.read_mesh_data()

    def read_mf_int(self, mesh, name: str = ""):
        """Read MeshFunction of type int"""
        return self._cpp_object.read_mf_int(mesh, name)

    def read_mvc_int(self, mesh, name: str = ""):
        """Read MeshFunction of type int"""
        return self._cpp_object.read_mvc_int(mesh, name)
