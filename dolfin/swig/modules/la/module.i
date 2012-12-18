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


// The PyDOLFIN extension module for the la module
%module(package="dolfin.cpp.la", directors="1") la

// Define module name for conditional includes
#define LAMODULE

%{

// Include types from dependent modules

// #include types from common submodule of module common
#include "dolfin/common/types.h"
#include "dolfin/common/Array.h"
#include "dolfin/common/Variable.h"

// #include types from parameter submodule of module common
#include "dolfin/parameter/Parameter.h"
#include "dolfin/parameter/Parameters.h"

// #include types from graph submodule of module mesh
#include "dolfin/graph/Graph.h"

// Include types from present module la

// #include types from la submodule
#include "dolfin/la/ublas.h"
#include "dolfin/la/LinearAlgebraObject.h"
#include "dolfin/la/GenericLinearOperator.h"
#include "dolfin/la/GenericTensor.h"
#include "dolfin/la/GenericMatrix.h"
#include "dolfin/la/GenericSparsityPattern.h"
#include "dolfin/la/GenericVector.h"
#include "dolfin/la/GenericLinearSolver.h"
#include "dolfin/la/GenericLUSolver.h"
#include "dolfin/la/GenericPreconditioner.h"
#include "dolfin/la/PETScObject.h"
#include "dolfin/la/PETScBaseMatrix.h"
#include "dolfin/la/uBLASFactory.h"
#include "dolfin/la/uBLASMatrix.h"
#include "dolfin/la/uBLASLinearOperator.h"
#include "dolfin/la/PETScMatrix.h"
#include "dolfin/la/PETScLinearOperator.h"
#include "dolfin/la/PETScPreconditioner.h"
#include "dolfin/la/EpetraLUSolver.h"
#include "dolfin/la/EpetraKrylovSolver.h"
#include "dolfin/la/EpetraMatrix.h"
#include "dolfin/la/EpetraVector.h"
#include "dolfin/la/PETScKrylovSolver.h"
#include "dolfin/la/PETScLUSolver.h"
#include "dolfin/la/CholmodCholeskySolver.h"
#include "dolfin/la/UmfpackLUSolver.h"
#include "dolfin/la/MUMPSLUSolver.h"
#include "dolfin/la/PaStiXLUSolver.h"
#include "dolfin/la/STLMatrix.h"
#include "dolfin/la/CoordinateMatrix.h"
#include "dolfin/la/uBLASVector.h"
#include "dolfin/la/PETScVector.h"
#include "dolfin/la/SparsityPattern.h"
#include "dolfin/la/GenericLinearAlgebraFactory.h"
#include "dolfin/la/DefaultFactory.h"
#include "dolfin/la/PETScUserPreconditioner.h"
#include "dolfin/la/PETScFactory.h"
#include "dolfin/la/PETScCuspFactory.h"
#include "dolfin/la/EpetraFactory.h"
#include "dolfin/la/STLFactory.h"
#include "dolfin/la/SLEPcEigenSolver.h"
#include "dolfin/la/TAOLinearBoundSolver.h"
#include "dolfin/la/TrilinosPreconditioner.h"
#include "dolfin/la/uBLASSparseMatrix.h"
#include "dolfin/la/uBLASDenseMatrix.h"
#include "dolfin/la/uBLASPreconditioner.h"
#include "dolfin/la/uBLASKrylovSolver.h"
#include "dolfin/la/uBLASILUPreconditioner.h"
#include "dolfin/la/Vector.h"
#include "dolfin/la/Matrix.h"
#include "dolfin/la/Scalar.h"
#include "dolfin/la/LinearSolver.h"
#include "dolfin/la/KrylovSolver.h"
#include "dolfin/la/LUSolver.h"
#include "dolfin/la/solve.h"
#include "dolfin/la/BlockVector.h"
#include "dolfin/la/BlockMatrix.h"
#include "dolfin/la/TensorProductVector.h"
#include "dolfin/la/TensorProductMatrix.h"
#include "dolfin/la/LinearOperator.h"

// #include types from nls submodule
#include "dolfin/nls/NonlinearProblem.h"
#include "dolfin/nls/NewtonSolver.h"
#include "dolfin/nls/PETScSNESSolver.h"

// NumPy includes
#define PY_ARRAY_UNIQUE_SYMBOL PyDOLFIN_LA
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

// %import types from submodule parameter of SWIG module common
%include "dolfin/swig/parameter/pre.i"
%import(module="common") "dolfin/parameter/Parameter.h"
%import(module="common") "dolfin/parameter/Parameters.h"

// %import types from submodule graph of SWIG module mesh
%import(module="mesh") "dolfin/graph/Graph.h"

// Turn on SWIG generated signature documentation and include doxygen
// generated docstrings
//%feature("autodoc", "1");
%include "dolfin/swig/la/docstrings.i"
%include "dolfin/swig/nls/docstrings.i"

// %include types from submodule la
%include "dolfin/swig/la/pre.i"
%include "dolfin/la/ublas.h"
%include "dolfin/la/LinearAlgebraObject.h"
%include "dolfin/la/GenericLinearOperator.h"
%include "dolfin/la/GenericTensor.h"
%include "dolfin/la/GenericMatrix.h"
%include "dolfin/la/GenericSparsityPattern.h"
%include "dolfin/la/GenericVector.h"
%include "dolfin/la/GenericLinearSolver.h"
%include "dolfin/la/GenericLUSolver.h"
%include "dolfin/la/GenericPreconditioner.h"
%include "dolfin/la/PETScObject.h"
%include "dolfin/la/PETScBaseMatrix.h"
%include "dolfin/la/uBLASFactory.h"
%include "dolfin/la/uBLASMatrix.h"
%include "dolfin/la/uBLASLinearOperator.h"
%include "dolfin/la/PETScMatrix.h"
%include "dolfin/la/PETScLinearOperator.h"
%include "dolfin/la/PETScPreconditioner.h"
%include "dolfin/la/EpetraLUSolver.h"
%include "dolfin/la/EpetraKrylovSolver.h"
%include "dolfin/la/EpetraMatrix.h"
%include "dolfin/la/EpetraVector.h"
%include "dolfin/la/PETScKrylovSolver.h"
%include "dolfin/la/PETScLUSolver.h"
%include "dolfin/la/CholmodCholeskySolver.h"
%include "dolfin/la/UmfpackLUSolver.h"
%include "dolfin/la/MUMPSLUSolver.h"
%include "dolfin/la/PaStiXLUSolver.h"
%include "dolfin/la/STLMatrix.h"
%include "dolfin/la/CoordinateMatrix.h"
%include "dolfin/la/uBLASVector.h"
%include "dolfin/la/PETScVector.h"
%include "dolfin/la/SparsityPattern.h"
%include "dolfin/la/GenericLinearAlgebraFactory.h"
%include "dolfin/la/DefaultFactory.h"
%include "dolfin/la/PETScUserPreconditioner.h"
%include "dolfin/la/PETScFactory.h"
%include "dolfin/la/PETScCuspFactory.h"
%include "dolfin/la/EpetraFactory.h"
%include "dolfin/la/STLFactory.h"
%include "dolfin/la/SLEPcEigenSolver.h"
%include "dolfin/la/TAOLinearBoundSolver.h"
%include "dolfin/la/TrilinosPreconditioner.h"
%include "dolfin/la/uBLASSparseMatrix.h"
%include "dolfin/la/uBLASDenseMatrix.h"
%include "dolfin/la/uBLASPreconditioner.h"
%include "dolfin/la/uBLASKrylovSolver.h"
%include "dolfin/la/uBLASILUPreconditioner.h"
%include "dolfin/la/Vector.h"
%include "dolfin/la/Matrix.h"
%include "dolfin/la/Scalar.h"
%include "dolfin/la/LinearSolver.h"
%include "dolfin/la/KrylovSolver.h"
%include "dolfin/la/LUSolver.h"
%include "dolfin/la/solve.h"
%include "dolfin/la/BlockVector.h"
%include "dolfin/la/BlockMatrix.h"
%include "dolfin/la/TensorProductVector.h"
%include "dolfin/la/TensorProductMatrix.h"
%include "dolfin/la/LinearOperator.h"
%include "dolfin/swig/la/post.i"

// %include types from submodule nls
%include "dolfin/swig/nls/pre.i"
%include "dolfin/nls/NonlinearProblem.h"
%include "dolfin/nls/NewtonSolver.h"
%include "dolfin/nls/PETScSNESSolver.h"

