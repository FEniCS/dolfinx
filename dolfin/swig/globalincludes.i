// Code snippets for global includes for the combined SWIG modules
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
// First added:  2012-01-17
// Last changed: 2012-01-18

%include <stdint.i>

// Global shared_ptr classes
// NOTE: These classes need to be declared for all classes in all combined
// NOTE: modules
%include "dolfin/swig/shared_ptr_classes.i"

// Global typemaps
%include "dolfin/swig/typemaps/includes.i"

// Fragments
%fragment("NoDelete", "header") {
%#include "dolfin/common/NoDeleter.h"
}

// Global exceptions
%include <exception.i>
%include "dolfin/swig/exceptions.i"

// Do not expand default arguments in C++ by generating two an extra
// function in the SWIG layer. This reduces code bloat.
// NOTE: Hake Commenting out compactdefaultargs as it creates problems for SWIG
// NOTE: to evaluate bool arguments with default values where another method with
// NOTE: the same number of arguments exists.
//%feature("compactdefaultargs");

// STL SWIG string class
%include <std_string.i>

// Include information about swig version
%include "dolfin/swig/version.i"

// Include SWIG defines
%include "dolfin/swig/defines.i"

// Include all forward declarations
%include "dolfin/swig/forwarddeclarations.i"
