# -*- coding: utf-8 -*-
# Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfin.cpp as cpp
import dolfin.function.function

# Functions to extend cpp.io.HDF5File with


def read_function(self, V, name):
    # Read cpp function
    u_cpp = self.read(V._cpp_object, name)
    return dolfin.function.function.Function(V, u_cpp.vector())


cpp.io.HDF5File.read_function = read_function
del read_function


def read_checkpoint(self, V, name, counter=-1):
    # Read cpp function
    u_cpp = self._read_checkpoint(V._cpp_object, name, counter)
    return dolfin.function.function.Function(V, u_cpp.vector())


cpp.io.XDMFFile.read_checkpoint = read_checkpoint

del read_checkpoint
