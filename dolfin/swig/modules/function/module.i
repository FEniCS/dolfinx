// Auto generated SWIG file for Python interface of DOLFIN
//
// Copyright (C) 2012 Johan Hake
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
// First added:  2012-01-18
// Last changed: 2012-02-01


// The PyDOLFIN extension module for the function module
%module(package="dolfin.cpp.function", directors="1") function

%{
#include <dolfin/dolfin.h>
#define PY_ARRAY_UNIQUE_SYMBOL PyDOLIN_FUNCTION
#include <numpy/arrayobject.h>
%}

%init%{
import_array();
%}

// Include global SWIG interface files:
// Typemaps, shared_ptr declarations, exceptions, version
%include "dolfin/swig/globalincludes.i"

// Import types from other combined modules
%include "dolfin/swig/common/local_imports.i"
%include "dolfin/swig/parameter/local_imports.i"
%include "dolfin/swig/log/local_imports.i"
%include "dolfin/swig/la/local_imports.i"
%include "dolfin/swig/nls/local_imports.i"
%include "dolfin/swig/intersection/local_imports.i"
%include "dolfin/swig/mesh/local_imports.i"
%include "dolfin/swig/refinement/local_imports.i"
%include "dolfin/swig/graph/local_imports.i"
%include "dolfin/swig/plot/local_imports.i"
%include "dolfin/swig/quadrature/local_imports.i"
%include "dolfin/swig/ale/local_imports.i"
%include "dolfin/swig/fem/local_imports.i"
%include "dolfin/swig/adaptivity/local_imports.i"
%include "dolfin/swig/io/local_imports.i"

// Turn on SWIG generated signature documentation and include doxygen
// generated docstrings
//%feature("autodoc", "1");
%include "dolfin/swig/function/docstrings.i"
%include "dolfin/swig/math/docstrings.i"

// Include generated include files for the DOLFIN headers for this module
%include "dolfin/swig/function/includes.i"
%include "dolfin/swig/math/includes.i"

