# Code snippets for autogenerations of SWIG code
#
# Copyright (C) 2012 Johan Hake
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2012-01-17
# Last changed: 2012-07-03

copyright_statement = r"""%(comment)s Auto generated SWIG file for Python interface of DOLFIN
%(comment)s
%(comment)s Copyright (C) 2012 %(holder)s
%(comment)s
%(comment)s This file is part of DOLFIN.
%(comment)s
%(comment)s DOLFIN is free software: you can redistribute it and/or modify
%(comment)s it under the terms of the GNU Lesser General Public License as published by
%(comment)s the Free Software Foundation, either version 3 of the License, or
%(comment)s (at your option) any later version.
%(comment)s
%(comment)s DOLFIN is distributed in the hope that it will be useful,
%(comment)s but WITHOUT ANY WARRANTY; without even the implied warranty of
%(comment)s MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
%(comment)s GNU Lesser General Public License for more details.
%(comment)s
%(comment)s You should have received a copy of the GNU Lesser General Public License
%(comment)s along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
%(comment)s

"""

# FIXME: Removed date from copyright form 
"""
%(comment)s First added:  2012-01-18
%(comment)s Last changed: %(year)d-%(month)0.2d-%(day)0.2d

"""

# Template code for all combined SWIG modules
module_template = r"""
// The PyDOLFIN extension module for the %(module)s module
%%module(package="dolfin.cpp.%(module)s", directors="1") %(module)s

// Define module name for conditional includes
#define %(MODULE)sMODULE

%%{
%(headers)s

// NumPy includes
#define PY_ARRAY_UNIQUE_SYMBOL PyDOLFIN_%(MODULE)s
#include <numpy/arrayobject.h>
%%}

%%init%%{
import_array();
%%}

// Include global SWIG interface files:
// Typemaps, shared_ptr declarations, exceptions, version
%%include "dolfin/swig/globalincludes.i"
%(imports)s

// Turn on SWIG generated signature documentation and include doxygen
// generated docstrings
//%%feature("autodoc", "1");
%(docstrings)s
%(includes)s

"""
