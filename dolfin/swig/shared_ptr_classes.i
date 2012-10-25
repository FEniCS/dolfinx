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
// Modified by Anders logg, 2009.
// Modified by Garth N. Wells, 2009.
// Modified by Fredrik Valdmanis, 2012.
//
// First added:  2007-11-25
// Last changed: 2012-06-21

//=============================================================================
// SWIG directives for the shared_ptr stored classes in PyDOLFIN
//
// Objects of these classes can then be passed to c++ functions
// demanding a boost::shared_ptr<type>
//=============================================================================

//-----------------------------------------------------------------------------
// Un-comment these lines to use std::tr1, only works with swig version >=1.3.37
//-----------------------------------------------------------------------------
//#define SWIG_SHARED_PTR_NAMESPACE std
//#define SWIG_SHARED_PTR_SUBNAMESPACE tr1

//-----------------------------------------------------------------------------
// Include macros for shared_ptr support
//-----------------------------------------------------------------------------
%include <boost_shared_ptr.i>

//-----------------------------------------------------------------------------
// define to make SWIG_SHARED_PTR_QNAMESPACE available in inlined C++ code
//-----------------------------------------------------------------------------
%{
#define SWIG_SHARED_PTR_QNAMESPACE boost
%}

//-----------------------------------------------------------------------------
// Make DOLFIN aware of the types defined in UFC
//-----------------------------------------------------------------------------
%shared_ptr(ufc::cell_integral)
%shared_ptr(ufc::dofmap)
%shared_ptr(ufc::finite_element)
%shared_ptr(ufc::function)
%shared_ptr(ufc::form)
%shared_ptr(ufc::exterior_facet_integral)
%shared_ptr(ufc::interior_facet_integral)
%import(module="ufc") "ufc.h"

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

// common
%shared_ptr(dolfin::Variable)

// fem
%shared_ptr(dolfin::Hierarchical<dolfin::Form>)
%shared_ptr(dolfin::GenericDofMap)
%shared_ptr(dolfin::DofMap)
%shared_ptr(dolfin::Form)
%shared_ptr(dolfin::FiniteElement)
%shared_ptr(dolfin::BasisFunction)

%shared_ptr(dolfin::Hierarchical<dolfin::LinearVariationalProblem>)
%shared_ptr(dolfin::Hierarchical<dolfin::NonlinearVariationalProblem>)
%shared_ptr(dolfin::LinearVariationalProblem)
%shared_ptr(dolfin::NonlinearVariationalProblem)
%shared_ptr(dolfin::LinearVariationalSolver)
%shared_ptr(dolfin::NonlinearVariationalSolver)
%shared_ptr(dolfin::VariationalProblem)

%shared_ptr(dolfin::BoundaryCondition)
%shared_ptr(dolfin::Hierarchical<dolfin::DirichletBC>)
%shared_ptr(dolfin::DirichletBC)
%shared_ptr(dolfin::PeriodicBC)

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

// mesh
%shared_ptr(dolfin::Hierarchical<dolfin::Mesh>)
%shared_ptr(dolfin::Mesh)
%shared_ptr(dolfin::BoundaryMesh)
%shared_ptr(dolfin::SubMesh)
%shared_ptr(dolfin::UnitTetrahedron)
%shared_ptr(dolfin::UnitCube)
%shared_ptr(dolfin::UnitInterval)
%shared_ptr(dolfin::Interval)
%shared_ptr(dolfin::UnitTriangle)
%shared_ptr(dolfin::UnitSquare)
%shared_ptr(dolfin::UnitCircle)
%shared_ptr(dolfin::Box)
%shared_ptr(dolfin::Rectangle)
%shared_ptr(dolfin::UnitSphere)

%shared_ptr(dolfin::SubDomain)
%shared_ptr(dolfin::DomainBoundary)

// mesh
%shared_ptr(dolfin::LocalMeshData)
%shared_ptr(dolfin::MeshData)

// NOTE: Most of the MeshFunctions are declared sharepointers in
// NOTE: mesh_pre.i, mesh_post.i
%shared_ptr(dolfin::Hierarchical<dolfin::MeshFunction<dolfin::uint> >)
%shared_ptr(dolfin::MeshFunction<dolfin::uint>)

// FIXME: Do we need to declare dolfin::uint?
%shared_ptr(dolfin::CellFunction<dolfin::uint>)
%shared_ptr(dolfin::EdgeFunction<dolfin::uint>)
%shared_ptr(dolfin::FaceFunction<dolfin::uint>)
%shared_ptr(dolfin::FacetFunction<dolfin::uint>)
%shared_ptr(dolfin::VertexFunction<dolfin::uint>)


// parameters
%shared_ptr(dolfin::Parameters)
%shared_ptr(dolfin::GlobalParameters)

// la
%shared_ptr(dolfin::GenericTensor)
%shared_ptr(dolfin::GenericVector)
%shared_ptr(dolfin::GenericMatrix)
%shared_ptr(dolfin::Scalar)

%shared_ptr(dolfin::Matrix)
%shared_ptr(dolfin::Vector)

%shared_ptr(dolfin::STLMatrix)
%shared_ptr(dolfin::uBLASMatrix<boost::numeric::ublas::matrix<double> >)
%shared_ptr(dolfin::uBLASMatrix<boost::numeric::ublas::compressed_matrix<double,\
            boost::numeric::ublas::row_major> >)
%shared_ptr(dolfin::uBLASVector)

#ifdef HAS_PETSC
%shared_ptr(dolfin::PETScBaseMatrix)
%shared_ptr(dolfin::PETScKrylovMatrix)
%shared_ptr(dolfin::PETScKrylovSolver)
%shared_ptr(dolfin::PETScLUSolver)
%shared_ptr(dolfin::PETScMatrix)
%shared_ptr(dolfin::PETScObject)
%shared_ptr(dolfin::PETScPreconditioner)
%shared_ptr(dolfin::PETScVector)
%shared_ptr(dolfin::PETScUserPreconditioner)
#endif

#ifdef HAS_SLEPC
%shared_ptr(dolfin::SLEPcEigenSolver)
#endif

#ifdef HAS_MTL4
%shared_ptr(dolfin::ITLKrylovSolver)
%shared_ptr(dolfin::MTL4Matrix)
%shared_ptr(dolfin::MTL4Vector)
#endif

#ifdef HAS_TRILINOS
%shared_ptr(dolfin::EpetraKrylovSolver)
%shared_ptr(dolfin::EpetraLUSolver)
%shared_ptr(dolfin::EpetraMatrix)
%shared_ptr(dolfin::EpetraSparsityPattern)
%shared_ptr(dolfin::EpetraVector)
%shared_ptr(dolfin::TrilinosPreconditioner)
#endif

#ifdef HAS_PASTIX
%shared_ptr(dolfin::PaStiXLUSolver)
#endif

%shared_ptr(dolfin::UmfpackLUSolver)
%shared_ptr(dolfin::CholmodCholeskySolver)

%shared_ptr(dolfin::uBLASKrylovSolver)

%shared_ptr(dolfin::LinearSolver)
%shared_ptr(dolfin::GenericLinearSolver)
%shared_ptr(dolfin::GenericLUSolver)
%shared_ptr(dolfin::KrylovSolver)
%shared_ptr(dolfin::LUSolver)
%shared_ptr(dolfin::SingularSolver)

%shared_ptr(dolfin::GenericSparsityPattern)
%shared_ptr(dolfin::SparsityPattern)

// log
%shared_ptr(dolfin::Table)

// math
%shared_ptr(dolfin::Lagrange)

// nls
%shared_ptr(dolfin::NewtonSolver)

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
