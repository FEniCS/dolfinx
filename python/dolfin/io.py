# -*- coding: utf-8 -*-
# Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfin.cpp as cpp
import . function.function

# Functions to extend cpp.io.HDF5File with

def read_function(self, V, name):
    # Read cpp function
    u_cpp = self.read(V._cpp_object, name)
    return function.function.Function(c_cpp)

cpp.io.HDF5File.read_function = read_function

del read_function

