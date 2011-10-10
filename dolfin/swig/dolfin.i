/* -*- C -*- */
// Copyright (C) 2005-2006 Johan Jansson
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders logg, 2005-2009.
// Modified by Ola Skavhaug, 2007-2009.
// Modified by Kent-Andre Mardal, 2008-2009.
// Modified by Johan Hake, 2008-2011.
// Modified by Garth N. Wells, 2009.
//
// First added:  2005-10-24
// Last changed: 2011-01-25

// The PyDOLFIN extension module
%module(package="dolfin", directors="1") cpp

%{
#include <dolfin/dolfin.h>
#define PY_ARRAY_UNIQUE_SYMBOL PyDOLFIN
#include <numpy/arrayobject.h>
%}

%init%{
import_array();
%}

// Global shared ptr declarations
%include "dolfin/swig/shared_ptr_classes.i"

// Global typemaps
%include "dolfin/swig/typemaps.i"
%include "dolfin/swig/std_pair_typemaps.i"
%include "dolfin/swig/numpy_typemaps.i"
%include "dolfin/swig/array_typemaps.i"
%include "dolfin/swig/std_vector_typemaps.i"
%include "dolfin/swig/std_set_typemaps.i"
%include "dolfin/swig/std_map_typemaps.i"

// Global exceptions
%include <exception.i>
%include "dolfin/swig/exceptions.i"

// Do not expand default arguments in C++ by generating two an extra 
// function in the SWIG layer. This reduces code bloat.
%feature("compactdefaultargs");

// STL SWIG string class
%include <std_string.i>

// Turn on SWIG generated signature documentation and include doxygen
// generated docstrings (Need to run generate.py to update the latter)
//%feature("autodoc", "1");
%include "dolfin/swig/docstrings.i"

// DOLFIN interface (Need to run generate.py to update this file)
%include "dolfin/swig/kernel_modules.i"

// Include information about swig version
%include "dolfin/swig/version.i"
