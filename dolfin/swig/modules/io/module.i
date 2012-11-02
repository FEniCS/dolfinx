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


// The PyDOLFIN extension module for the io module
%module(package="dolfin.cpp.io", directors="1") io

// Define module name for conditional includes
#define IOMODULE

%{

// Include types from dependent modules

// #include types from common submodule of module common
#include "dolfin/common/types.h"
#include "dolfin/common/Array.h"
#include "dolfin/common/Variable.h"
#include "dolfin/common/Hierarchical.h"

// #include types from parameter submodule of module common
#include "dolfin/parameter/Parameter.h"
#include "dolfin/parameter/Parameters.h"

// #include types from la submodule of module la
#include "dolfin/la/LinearAlgebraObject.h"
#include "dolfin/la/GenericLinearOperator.h"
#include "dolfin/la/GenericTensor.h"
#include "dolfin/la/GenericMatrix.h"
#include "dolfin/la/GenericVector.h"
#include "dolfin/la/Vector.h"
#include "dolfin/la/Matrix.h"

// #include types from mesh submodule of module mesh
#include "dolfin/mesh/MeshData.h"
#include "dolfin/mesh/Mesh.h"
#include "dolfin/mesh/MeshFunction.h"
#include "dolfin/mesh/MeshValueCollection.h"
#include "dolfin/mesh/LocalMeshData.h"

// #include types from function submodule of module function
#include "dolfin/function/GenericFunction.h"
#include "dolfin/function/Expression.h"
#include "dolfin/function/Function.h"
#include "dolfin/function/FunctionSpace.h"

// #include types from fem submodule of module fem
#include "dolfin/fem/GenericDofMap.h"
#include "dolfin/fem/DofMap.h"
#include "dolfin/fem/BoundaryCondition.h"
#include "dolfin/fem/DirichletBC.h"

// Include types from present module io

// #include types from plot submodule
#include "dolfin/plot/plot.h"
#include "dolfin/plot/VTKPlotter.h"

// #include types from io submodule
#include "dolfin/io/GenericFile.h"
#include "dolfin/io/File.h"
#include "dolfin/io/XDMFFile.h"
#include "dolfin/io/HDF5File.h"

// NumPy includes
#define PY_ARRAY_UNIQUE_SYMBOL PyDOLFIN_IO
#include <numpy/arrayobject.h>
%}

%init%{
import_array();
%}

// Include global SWIG interface files:
// Typemaps, shared_ptr declarations, exceptions, version
%include "dolfin/swig/globalincludes.i"

// %import types from submodule common of SWIG module common
%include "dolfin/swig/common/pre.i"
%import(module="common") "dolfin/common/types.h"
%import(module="common") "dolfin/common/Array.h"
%import(module="common") "dolfin/common/Variable.h"
%import(module="common") "dolfin/common/Hierarchical.h"

// %import types from submodule parameter of SWIG module common
%include "dolfin/swig/parameter/pre.i"
%import(module="common") "dolfin/parameter/Parameter.h"
%import(module="common") "dolfin/parameter/Parameters.h"

// %import types from submodule la of SWIG module la
%include "dolfin/swig/la/pre.i"
%import(module="la") "dolfin/la/LinearAlgebraObject.h"
%import(module="la") "dolfin/la/GenericLinearOperator.h"
%import(module="la") "dolfin/la/GenericTensor.h"
%import(module="la") "dolfin/la/GenericMatrix.h"
%import(module="la") "dolfin/la/GenericVector.h"
%import(module="la") "dolfin/la/Vector.h"
%import(module="la") "dolfin/la/Matrix.h"

// %import types from submodule mesh of SWIG module mesh
%include "dolfin/swig/mesh/pre.i"
%import(module="mesh") "dolfin/mesh/MeshData.h"
%import(module="mesh") "dolfin/mesh/Mesh.h"
%import(module="mesh") "dolfin/mesh/MeshFunction.h"
%import(module="mesh") "dolfin/mesh/MeshValueCollection.h"
%import(module="mesh") "dolfin/mesh/LocalMeshData.h"

// %import types from submodule function of SWIG module function
%include "dolfin/swig/function/pre.i"
%import(module="function") "dolfin/function/GenericFunction.h"
%import(module="function") "dolfin/function/Expression.h"
%import(module="function") "dolfin/function/Function.h"
%import(module="function") "dolfin/function/FunctionSpace.h"

// %import types from submodule fem of SWIG module fem
%include "dolfin/swig/fem/pre.i"
%import(module="fem") "dolfin/fem/GenericDofMap.h"
%import(module="fem") "dolfin/fem/DofMap.h"
%import(module="fem") "dolfin/fem/BoundaryCondition.h"
%import(module="fem") "dolfin/fem/DirichletBC.h"

// Turn on SWIG generated signature documentation and include doxygen
// generated docstrings
//%feature("autodoc", "1");
%include "dolfin/swig/plot/docstrings.i"
%include "dolfin/swig/io/docstrings.i"

// %include types from submodule plot
%include "dolfin/plot/plot.h"
%include "dolfin/plot/VTKPlotter.h"

// %include types from submodule io
%include "dolfin/swig/io/pre.i"
%include "dolfin/io/GenericFile.h"
%include "dolfin/io/File.h"
%include "dolfin/io/XDMFFile.h"
%include "dolfin/io/HDF5File.h"
%include "dolfin/swig/io/post.i"

