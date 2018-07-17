# -*- coding: utf-8 -*-
# Copyright (C) 2018 Chris N. Richardson, Garth N. Wells, Michal Habera
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfin.cpp as cpp
from dolfin.function.functionspace import FunctionSpace
from dolfin.function.function import Function


class HDF5File(cpp.io.HDF5File):

    def read_function(self, V: FunctionSpace, name: str):
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
        if not isinstance(V, FunctionSpace):
            raise TypeError("expected dolfin.FunctionSpace as first argument")

        u_cpp = self.read(V._cpp_object, name)
        return Function(V, u_cpp.vector())
