/* -*- C -*- */
// Copyright (C) 2007-2012 Johan Hake
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
// Modified by Anders Logg, 2009.
// Modified by Garth N. Wells, 2009-2012.
// Modified by Fredrik Valdmanis, 2012.
// Modified by Patrick E. Farrell, 2012.
// Modified by Benjamin Kehlet, 2012.
// Modified by Andre Massing, 2013.
// Modified by Tianyi Li, 2014.
//
// First added:  2007-11-25
// Last changed: 2014-08-11

//=============================================================================
// SWIG directives for the shared_ptr stored classes in PyDOLFIN
//
// Objects of these classes can then be passed to c++ functions
// demanding a std::shared_ptr<type>
//=============================================================================

//-----------------------------------------------------------------------------
// Include macros for shared_ptr support
//-----------------------------------------------------------------------------
%include <std_shared_ptr.i>

//-----------------------------------------------------------------------------
// Declare shared_ptr stored types in UFC
//-----------------------------------------------------------------------------
%shared_ptr(ufc::function)
%shared_ptr(ufc::dofmap)
%shared_ptr(ufc::finite_element)
%shared_ptr(ufc::form)

//-----------------------------------------------------------------------------
// Declare shared_ptr stored types in PyDOLFIN
//-----------------------------------------------------------------------------

// adaptivity
%shared_ptr(dolfin::AdaptiveLinearVariationalSolver)
%shared_ptr(dolfin::AdaptiveNonlinearVariationalSolver)
%shared_ptr(dolfin::ErrorControl)
%shared_ptr(dolfin::Hierarchical<dolfin::ErrorControl>)
%shared_ptr(dolfin::GenericAdaptiveVariationalSolver)
%shared_ptr(dolfin::GoalFunctional)
%shared_ptr(dolfin::SpecialFacetFunction)
%shared_ptr(dolfin::TimeSeries)
%shared_ptr(dolfin::TimeSeriesHDF5)

// ale
%shared_ptr(dolfin::MeshDisplacement)

// common
%shared_ptr(dolfin::Variable)

// fem
%shared_ptr(dolfin::Hierarchical<dolfin::Form>)
%shared_ptr(dolfin::GenericDofMap)
%shared_ptr(dolfin::DofMap)
%shared_ptr(dolfin::MultiMeshDofMap)
%shared_ptr(dolfin::Form)
%shared_ptr(dolfin::FiniteElement)
%shared_ptr(dolfin::BasisFunction)
%shared_ptr(dolfin::MultiStageScheme)

%shared_ptr(dolfin::Hierarchical<dolfin::LinearVariationalProblem>)
%shared_ptr(dolfin::Hierarchical<dolfin::NonlinearVariationalProblem>)
%shared_ptr(dolfin::LinearVariationalProblem)
%shared_ptr(dolfin::NonlinearVariationalProblem)
%shared_ptr(dolfin::LinearVariationalSolver)
%shared_ptr(dolfin::NonlinearVariationalSolver)
%shared_ptr(dolfin::PointIntegralSolver)

%shared_ptr(dolfin::Hierarchical<dolfin::DirichletBC>)
%shared_ptr(dolfin::DirichletBC)

// function
%shared_ptr(dolfin::Hierarchical<dolfin::FunctionSpace>)
%shared_ptr(dolfin::FunctionSpace)
%shared_ptr(dolfin::SubSpace)

%shared_ptr(dolfin::GenericFunction)
%shared_ptr(dolfin::Hierarchical<dolfin::Function>)
%shared_ptr(dolfin::Function)
%shared_ptr(dolfin::Expression)
%shared_ptr(dolfin::FacetArea)
%shared_ptr(dolfin::Constant)
%shared_ptr(dolfin::MeshCoordinates)
%shared_ptr(dolfin::MultiMeshFunctionSpace)
%shared_ptr(dolfin::MultiMeshSubSpace)

// geometry
%shared_ptr(dolfin::BoundingBoxTree)

// mesh
%shared_ptr(dolfin::Hierarchical<dolfin::Mesh>)
%shared_ptr(dolfin::BoundaryMesh)
%shared_ptr(dolfin::Mesh)
%shared_ptr(dolfin::SubMesh)
%shared_ptr(dolfin::UnitTetrahedronMesh)
%shared_ptr(dolfin::UnitCubeMesh)
%shared_ptr(dolfin::UnitIntervalMesh)
%shared_ptr(dolfin::IntervalMesh)
%shared_ptr(dolfin::UnitTriangleMesh)
%shared_ptr(dolfin::UnitSquareMesh)
%shared_ptr(dolfin::UnitQuadMesh)
%shared_ptr(dolfin::UnitHexMesh)
%shared_ptr(dolfin::UnitDiscMesh)
%shared_ptr(dolfin::SphericalShellMesh)
%shared_ptr(dolfin::BoxMesh)
%shared_ptr(dolfin::Box)
%shared_ptr(dolfin::RectangleMesh)
%shared_ptr(dolfin::Rectangle)
%shared_ptr(dolfin::MultiMesh)
%shared_ptr(dolfin::SubDomain)
%shared_ptr(dolfin::DomainBoundary)
%shared_ptr(dolfin::LocalMeshData)
%shared_ptr(dolfin::MeshData)
%shared_ptr(dolfin::MeshHierarchy)

// NOTE: Most of the MeshFunctions are declared shared pointers in
// NOTE: mesh/pre.i, mesh/post.i
%shared_ptr(dolfin::Hierarchical<dolfin::MeshFunction<std::size_t> >)
%shared_ptr(dolfin::MeshFunction<std::size_t>)

%shared_ptr(dolfin::CellFunction<std::size_t>)
%shared_ptr(dolfin::EdgeFunction<std::size_t>)
%shared_ptr(dolfin::FaceFunction<std::size_t>)
%shared_ptr(dolfin::FacetFunction<std::size_t>)
%shared_ptr(dolfin::VertexFunction<std::size_t>)

// parameters
%shared_ptr(dolfin::Parameters)
%shared_ptr(dolfin::GlobalParameters)

// la
%shared_ptr(dolfin::GenericLinearOperator)
%shared_ptr(dolfin::GenericMatrix)
%shared_ptr(dolfin::GenericPreconditioner)
%shared_ptr(dolfin::GenericTensor)
%shared_ptr(dolfin::GenericVector)
%shared_ptr(dolfin::LinearAlgebraObject)
%shared_ptr(dolfin::Scalar)

%shared_ptr(dolfin::Matrix)
%shared_ptr(dolfin::Vector)
%shared_ptr(dolfin::LinearOperator)
%shared_ptr(dolfin::IndexMap)

%shared_ptr(dolfin::STLMatrix)

%shared_ptr(dolfin::EigenKrylovSolver)
%shared_ptr(dolfin::EigenLUSolver)
%shared_ptr(dolfin::EigenMatrix)
%shared_ptr(dolfin::EigenVector)

#ifdef HAS_PETSC
%shared_ptr(dolfin::PETScBaseMatrix)
%shared_ptr(dolfin::PETScLinearOperator)
%shared_ptr(dolfin::PETScKrylovSolver)
%shared_ptr(dolfin::PETScLUSolver)
%shared_ptr(dolfin::PETScMatrix)
%shared_ptr(dolfin::PETScObject)
%shared_ptr(dolfin::PETScPreconditioner)
%shared_ptr(dolfin::PETScVector)
%shared_ptr(dolfin::PETScUserPreconditioner)
#endif

#ifdef HAS_TRILINOS
%shared_ptr(dolfin::TpetraVector)
%shared_ptr(dolfin::TpetraMatrix)
%shared_ptr(dolfin::BelosKrylovSolver)
%shared_ptr(dolfin::TrilinosPreconditioner)
%shared_ptr(dolfin::Ifpack2Preconditioner)
%shared_ptr(dolfin::MueluPreconditioner)
#endif

#ifdef HAS_SLEPC
%shared_ptr(dolfin::SLEPcEigenSolver)
#endif

#ifdef HAS_PASTIX
%shared_ptr(dolfin::PaStiXLUSolver)
#endif

%shared_ptr(dolfin::UmfpackLUSolver)
%shared_ptr(dolfin::CholmodCholeskySolver)

%shared_ptr(dolfin::LinearSolver)
%shared_ptr(dolfin::GenericLinearSolver)
%shared_ptr(dolfin::GenericLUSolver)
%shared_ptr(dolfin::KrylovSolver)
%shared_ptr(dolfin::LUSolver)

%shared_ptr(dolfin::TensorLayout)
%shared_ptr(dolfin::SparsityPattern)

// log
%shared_ptr(dolfin::Table)

// io
%shared_ptr(dolfin::GenericFile)
%shared_ptr(dolfin::File)
%shared_ptr(dolfin::XDMFFile)
%shared_ptr(dolfin::HDF5File)

// math
%shared_ptr(dolfin::Lagrange)

// nls
%shared_ptr(dolfin::NewtonSolver)
%shared_ptr(dolfin::PETScSNESSolver)
#ifdef HAS_PETSC
%shared_ptr(dolfin::TAOLinearBoundSolver)
%shared_ptr(dolfin::PETScTAOSolver)
#endif
// plot
%shared_ptr(dolfin::VTKPlotter)
%shared_ptr(dolfin::GenericVTKPlottable)
%shared_ptr(dolfin::VTKPlottableMesh)
%shared_ptr(dolfin::VTKPlottableGenericFunction)
%shared_ptr(dolfin::VTKPlottableMeshFunction)
%shared_ptr(dolfin::ExpressionWrapper)

// quadrature
%shared_ptr(dolfin::Quadrature)
%shared_ptr(dolfin::LobattoQuadrature)
%shared_ptr(dolfin::RadauQuadrature)
%shared_ptr(dolfin::GaussQuadrature)
%shared_ptr(dolfin::GaussianQuadrature)

// geometry
%shared_ptr(dolfin::MeshPointIntersection)
