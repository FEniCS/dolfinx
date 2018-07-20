# -*- coding: utf-8 -*-
# Copyright (C) 2017 Chris N. Richardson, Garth N. Wells, Michal Habera
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""IO interfaces for inout data, post-processing and checkpointing"""

import dolfin.cpp as cpp
from dolfin.function.function import Function


class HDF5File(cpp.io.HDF5File):
    """Interface to HDF5 files"""

    def read_function(self, V, name: str):
        """Read finite element Function from file

        Parameters
        ----------
        V
            Function space of saved function.
        name
            Name of function as saved into HDF file.
        Note
        ----
        Parameter `V: Function space` must be the same as saved function space
        except for ordering of mesh entities.
        Returns
        -------
        dolfin.function.function.Function
            Function read from file
        """

        V_cpp = getattr(V, "_cpp_object", V)
        u_cpp = self.read(V_cpp, name)
        return Function(V, u_cpp.vector())


class VTKFile(cpp.io.VTKFile):
    """Interface to VTK files"""
    pass


class XDMFFile(cpp.io.XDMFFile):
    """Interface to XDMF files"""

    def read_checkpoint(self, V, name: str, counter: int=-1):
        """Read finite element Function from checkpointing format
        Parameters
        ----------
        V
            Function space of saved function.
        name
            Name of function as saved into XDMF file.
        counter : optional
            Position of function in the file within functions of the same
            name. Counter is used to read function saved as time series.
            To get last saved function use counter=-1, or counter=-2 for
            one before last, etc.
        Note
        ----
        Parameter `V: Function space` must be the same as saved function space
        except for ordering of mesh entities.
        Returns
        -------
        dolfin.function.function.Function
            The finite element Function read from checkpoint file
        """

        V_cpp = getattr(V, "_cpp_object", V)
        u_cpp = self._read_checkpoint(V_cpp, name, counter)
        return Function(V, u_cpp.vector())
