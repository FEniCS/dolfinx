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
// Last changed: 2012-03-02


// The PyDOLFIN extension module for the mesh module
%module(package="dolfin.cpp.mesh", directors="1") mesh

%{
#include <dolfin/dolfin.h>
#define PY_ARRAY_UNIQUE_SYMBOL PyDOLIN_MESH
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
%include "dolfin/swig/function/local_imports.i"
%include "dolfin/swig/plot/local_imports.i"
%include "dolfin/swig/math/local_imports.i"
%include "dolfin/swig/quadrature/local_imports.i"
%include "dolfin/swig/fem/local_imports.i"
%include "dolfin/swig/adaptivity/local_imports.i"
%include "dolfin/swig/io/local_imports.i"

// Turn on SWIG generated signature documentation and include doxygen
// generated docstrings
//%feature("autodoc", "1");
%include "dolfin/swig/intersection/docstrings.i"
%include "dolfin/swig/mesh/docstrings.i"
%include "dolfin/swig/generation/docstrings.i"
%include "dolfin/swig/refinement/docstrings.i"
%include "dolfin/swig/graph/docstrings.i"
%include "dolfin/swig/ale/docstrings.i"

// Include generated include files for the DOLFIN headers for this module
%include "dolfin/swig/intersection/includes.i"
%include "dolfin/swig/mesh/includes.i"
%include "dolfin/swig/generation/includes.i"
%include "dolfin/swig/refinement/includes.i"
%include "dolfin/swig/graph/includes.i"
%include "dolfin/swig/ale/includes.i"

