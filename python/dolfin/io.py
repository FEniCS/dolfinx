# -*- coding: utf-8 -*-
# Copyright (C) 2017 Chris N. Richardson, Garth N. Wells, Michal Habera
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfin.cpp as cpp
from dolfin.function.function import Function


class HDF5File(cpp.io.HDF5File):

    def read_function(self, V, name: str):
        """Reads in function

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
        dolfin.Function
        """

        V_cpp = getattr(V, "_cpp_object", V)
        u_cpp = self.read(V_cpp, name)
        return Function(V, u_cpp.vector())


class VTKFile(cpp.io.VTKFile):
    pass


class XDMFFile(cpp.io.XDMFFile):

    def read_checkpoint(self, V, name: str, counter: int=-1):
        """Reads in function from checkpointing format
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
        dolfin.Function
        """

        V_cpp = getattr(V, "_cpp_object", V)
        u_cpp = self._read_checkpoint(V_cpp, name, counter)
        return Function(V, u_cpp.vector())
