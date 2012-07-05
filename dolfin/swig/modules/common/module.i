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


// The PyDOLFIN extension module for the common module
%module(package="dolfin.cpp.common", directors="1") common

// Define module name for conditional includes
#define COMMONMODULE

%{

// Include types from dependent modules

// Include types from present module common

// #include types from common submodule
#include "dolfin/common/init.h"
#include "dolfin/common/defines.h"
#include "dolfin/common/types.h"
#include "dolfin/common/constants.h"
#include "dolfin/common/timing.h"
#include "dolfin/common/Array.h"
#include "dolfin/common/IndexSet.h"
#include "dolfin/common/Set.h"
#include "dolfin/common/Timer.h"
#include "dolfin/common/Variable.h"
#include "dolfin/common/Hierarchical.h"
#include "dolfin/common/MPI.h"
#include "dolfin/common/SubSystemsManager.h"

// #include types from parameter submodule
#include "dolfin/parameter/Parameter.h"
#include "dolfin/parameter/Parameters.h"
#include "dolfin/parameter/GlobalParameters.h"

// #include types from log submodule
#include "dolfin/log/log.h"
#include "dolfin/log/Event.h"
#include "dolfin/log/Progress.h"
#include "dolfin/log/Table.h"
#include "dolfin/log/LogLevel.h"

// NumPy includes
#define PY_ARRAY_UNIQUE_SYMBOL PyDOLFIN_COMMON
#include <numpy/arrayobject.h>
%}

%init%{
import_array();
%}

// Include global SWIG interface files:
// Typemaps, shared_ptr declarations, exceptions, version
%include "dolfin/swig/globalincludes.i"


// Turn on SWIG generated signature documentation and include doxygen
// generated docstrings
//%feature("autodoc", "1");
%include "dolfin/swig/common/docstrings.i"
%include "dolfin/swig/parameter/docstrings.i"
%include "dolfin/swig/log/docstrings.i"

// %include types from submodule common
%include "dolfin/swig/common/pre.i"
%include "dolfin/common/init.h"
%include "dolfin/common/defines.h"
%include "dolfin/common/types.h"
%include "dolfin/common/constants.h"
%include "dolfin/common/timing.h"
%include "dolfin/common/Array.h"
%include "dolfin/common/IndexSet.h"
%include "dolfin/common/Set.h"
%include "dolfin/common/Timer.h"
%include "dolfin/common/Variable.h"
%include "dolfin/common/Hierarchical.h"
%include "dolfin/common/MPI.h"
%include "dolfin/common/SubSystemsManager.h"
%include "dolfin/swig/common/post.i"

// %include types from submodule parameter
%include "dolfin/swig/parameter/pre.i"
%include "dolfin/parameter/Parameter.h"
%include "dolfin/parameter/Parameters.h"
%include "dolfin/parameter/GlobalParameters.h"
%include "dolfin/swig/parameter/post.i"

// %include types from submodule log
%include "dolfin/swig/log/pre.i"
%include "dolfin/log/log.h"
%include "dolfin/log/Event.h"
%include "dolfin/log/Progress.h"
%include "dolfin/log/Table.h"
%include "dolfin/log/LogLevel.h"
%include "dolfin/swig/log/post.i"

