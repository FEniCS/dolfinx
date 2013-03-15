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


// The PyDOLFIN extension module for the fem module
%module(package="dolfin.cpp.fem", directors="1") fem

// Define module name for conditional includes
#define FEMMODULE

%{

// Include types from dependent modules

// #include types from common submodule of module common
#include "dolfin/common/types.h"
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
#include "dolfin/la/GenericSparsityPattern.h"
#include "dolfin/la/GenericVector.h"
#include "dolfin/la/SparsityPattern.h"
#include "dolfin/la/Vector.h"
#include "dolfin/la/Matrix.h"

// #include types from mesh submodule of module mesh
#include "dolfin/mesh/Mesh.h"
#include "dolfin/mesh/MeshEntity.h"
#include "dolfin/mesh/Point.h"
#include "dolfin/mesh/Face.h"
#include "dolfin/mesh/Facet.h"
#include "dolfin/mesh/Cell.h"
#include "dolfin/mesh/MeshFunction.h"
#include "dolfin/mesh/SubDomain.h"
#include "dolfin/mesh/Restriction.h"

// #include types from function submodule of module function
#include "dolfin/function/GenericFunction.h"
#include "dolfin/function/Expression.h"
#include "dolfin/function/Function.h"
#include "dolfin/function/FunctionSpace.h"
#include "dolfin/function/SpecialFacetFunction.h"

// Include types from present module fem

// #include types from quadrature submodule
#include "dolfin/quadrature/BarycenterQuadrature.h"

// #include types from fem submodule
#include "dolfin/fem/GenericDofMap.h"
#include "dolfin/fem/DofMap.h"
#include "dolfin/fem/Equation.h"
#include "dolfin/fem/FiniteElement.h"
#include "dolfin/fem/BasisFunction.h"
#include "dolfin/fem/DirichletBC.h"
#include "dolfin/fem/PointSource.h"
#include "dolfin/fem/assemble.h"
#include "dolfin/fem/LocalSolver.h"
#include "dolfin/fem/solve.h"
#include "dolfin/fem/Form.h"
#include "dolfin/fem/AssemblerBase.h"
#include "dolfin/fem/Assembler.h"
#include "dolfin/fem/SparsityPatternBuilder.h"
#include "dolfin/fem/SystemAssembler.h"
#include "dolfin/fem/LinearVariationalProblem.h"
#include "dolfin/fem/LinearVariationalSolver.h"
#include "dolfin/fem/NonlinearVariationalProblem.h"
#include "dolfin/fem/NonlinearVariationalSolver.h"
#include "dolfin/fem/OpenMpAssembler.h"
#include "dolfin/fem/VariationalProblem.h"

// #include types from adaptivity submodule
#include "dolfin/adaptivity/GenericAdaptiveVariationalSolver.h"
#include "dolfin/adaptivity/AdaptiveLinearVariationalSolver.h"
#include "dolfin/adaptivity/AdaptiveNonlinearVariationalSolver.h"
#include "dolfin/adaptivity/GoalFunctional.h"
#include "dolfin/adaptivity/ErrorControl.h"
#include "dolfin/adaptivity/Extrapolation.h"
#include "dolfin/adaptivity/LocalAssembler.h"
#include "dolfin/adaptivity/TimeSeries.h"
#include "dolfin/adaptivity/adapt.h"
#include "dolfin/adaptivity/marking.h"
#include "dolfin/adaptivity/adaptivesolve.h"

// NumPy includes
#define PY_ARRAY_UNIQUE_SYMBOL PyDOLFIN_FEM
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
%import(module="la") "dolfin/la/GenericSparsityPattern.h"
%import(module="la") "dolfin/la/GenericVector.h"
%import(module="la") "dolfin/la/SparsityPattern.h"
%import(module="la") "dolfin/la/Vector.h"
%import(module="la") "dolfin/la/Matrix.h"

// %import types from submodule mesh of SWIG module mesh
%include "dolfin/swig/mesh/pre.i"
%import(module="mesh") "dolfin/mesh/Mesh.h"
%import(module="mesh") "dolfin/mesh/MeshEntity.h"
%import(module="mesh") "dolfin/mesh/Point.h"
%import(module="mesh") "dolfin/mesh/Face.h"
%import(module="mesh") "dolfin/mesh/Facet.h"
%import(module="mesh") "dolfin/mesh/Cell.h"
%import(module="mesh") "dolfin/mesh/MeshFunction.h"
%import(module="mesh") "dolfin/mesh/SubDomain.h"
%import(module="mesh") "dolfin/mesh/Restriction.h"

// %import types from submodule function of SWIG module function
%include "dolfin/swig/function/pre.i"
%import(module="function") "dolfin/function/GenericFunction.h"
%import(module="function") "dolfin/function/Expression.h"
%import(module="function") "dolfin/function/Function.h"
%import(module="function") "dolfin/function/FunctionSpace.h"
%import(module="function") "dolfin/function/SpecialFacetFunction.h"

// Turn on SWIG generated signature documentation and include doxygen
// generated docstrings
//%feature("autodoc", "1");
%include "dolfin/swig/quadrature/docstrings.i"
%include "dolfin/swig/fem/docstrings.i"
%include "dolfin/swig/adaptivity/docstrings.i"

// %include types from submodule quadrature
%include "dolfin/quadrature/BarycenterQuadrature.h"

// %include types from submodule fem
%include "dolfin/swig/fem/pre.i"
%include "dolfin/fem/GenericDofMap.h"
%include "dolfin/fem/DofMap.h"
%include "dolfin/fem/Equation.h"
%include "dolfin/fem/FiniteElement.h"
%include "dolfin/fem/BasisFunction.h"
%include "dolfin/fem/DirichletBC.h"
%include "dolfin/fem/PointSource.h"
%include "dolfin/fem/assemble.h"
%include "dolfin/fem/LocalSolver.h"
%include "dolfin/fem/solve.h"
%include "dolfin/fem/Form.h"
%include "dolfin/fem/AssemblerBase.h"
%include "dolfin/fem/Assembler.h"
%include "dolfin/fem/SparsityPatternBuilder.h"
%include "dolfin/fem/SystemAssembler.h"
%include "dolfin/fem/LinearVariationalProblem.h"
%include "dolfin/fem/LinearVariationalSolver.h"
%include "dolfin/fem/NonlinearVariationalProblem.h"
%include "dolfin/fem/NonlinearVariationalSolver.h"
%include "dolfin/fem/OpenMpAssembler.h"
%include "dolfin/fem/VariationalProblem.h"
%include "dolfin/swig/fem/post.i"

// %include types from submodule adaptivity
%include "dolfin/swig/adaptivity/pre.i"
%include "dolfin/adaptivity/GenericAdaptiveVariationalSolver.h"
%include "dolfin/adaptivity/AdaptiveLinearVariationalSolver.h"
%include "dolfin/adaptivity/AdaptiveNonlinearVariationalSolver.h"
%include "dolfin/adaptivity/GoalFunctional.h"
%include "dolfin/adaptivity/ErrorControl.h"
%include "dolfin/adaptivity/Extrapolation.h"
%include "dolfin/adaptivity/LocalAssembler.h"
%include "dolfin/adaptivity/TimeSeries.h"
%include "dolfin/adaptivity/adapt.h"
%include "dolfin/adaptivity/marking.h"
%include "dolfin/adaptivity/adaptivesolve.h"
%include "dolfin/swig/adaptivity/post.i"

