/* -*- C -*- */
// Copyright (C) 2009 Anders Logg
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
// Modified by Garth Wells, 2007-2011.
// Modified by Kent-Andre Mardal, 2008.
// Modified by Johan Hake, 2008-2009.
//
// First added:  2007-08-16
// Last changed: 2011-03-10

// ===========================================================================
// SWIG directives for the DOLFIN fem kernel module (pre)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
// ===========================================================================

//-----------------------------------------------------------------------------
// To simplify handling of shared_ptr types in PyDOLFIN we ignore the reference
// version of constructors to these types
//-----------------------------------------------------------------------------
%ignore dolfin::DirichletBC::DirichletBC(const FunctionSpace&,
					 const GenericFunction&,
					 const SubDomain&,
					 std::string method="topological");

%ignore dolfin::DirichletBC::DirichletBC(const FunctionSpace&,
					 const GenericFunction&,
					 const MeshFunction<unsigned int>&,
					 uint,
					 std::string method="topological");

%ignore dolfin::DirichletBC::DirichletBC(const FunctionSpace&,
					 const GenericFunction&,
					 uint,
					 std::string method="topological");

%ignore dolfin::DirichletBC::DirichletBC(boost::shared_ptr<const FunctionSpace>,
					 boost::shared_ptr<const GenericFunction>,
					 const std::vector<std::pair<uint, uint> >&,
					 std::string method="topological");

%ignore dolfin::PeriodicBC::PeriodicBC(const FunctionSpace&,
				       const SubDomain&);

%ignore dolfin::LinearVariationalProblem::LinearVariationalProblem(const Form&,
                                                                   const Form&,
                                                                   Function&);

%ignore dolfin::LinearVariationalProblem::LinearVariationalProblem(const Form&,
                                                                   const Form&,
                                                                   Function&,
                                                                   const BoundaryCondition&);

%ignore dolfin::LinearVariationalProblem::LinearVariationalProblem(const Form&,
                                                                   const Form&,
                                                                   Function&,
                                                                   std::vector<const BoundaryCondition*>);

%ignore dolfin::NonlinearVariationalProblem::NonlinearVariationalProblem(const Form&,
                                                                         Function&);

%ignore dolfin::NonlinearVariationalProblem::NonlinearVariationalProblem(const Form&,
                                                                         Function&,
                                                                         const Form&);

%ignore dolfin::NonlinearVariationalProblem::NonlinearVariationalProblem(const Form&,
                                                                         Function&,
                                                                         const BoundaryCondition&);

%ignore dolfin::NonlinearVariationalProblem::NonlinearVariationalProblem(const Form&,
                                                                         Function&,
                                                                         const BoundaryCondition&,
                                                                         const Form&);

%ignore dolfin::NonlinearVariationalProblem::NonlinearVariationalProblem(const Form&,
                                                                         Function&,
                                                                         std::vector<const BoundaryCondition*>);

%ignore dolfin::NonlinearVariationalProblem::NonlinearVariationalProblem(const Form&,
                                                                         Function&,
                                                                         std::vector<const BoundaryCondition*>,
                                                                         const Form&);

%ignore dolfin::LinearVariationalSolver::LinearVariationalSolver(LinearVariationalProblem&);

%ignore dolfin::NonlinearVariationalSolver::NonlinearVariationalSolver(NonlinearVariationalProblem&);

//-----------------------------------------------------------------------------
// Ignore operator= for DirichletBC to avoid warning
//-----------------------------------------------------------------------------
%ignore dolfin::DirichletBC::operator=;

//-----------------------------------------------------------------------------
// Ignore one of the constructors for DofMap to avoid warning
//-----------------------------------------------------------------------------
%ignore dolfin::DofMap::DofMap(boost::shared_ptr<const ufc::dofmap>, const Mesh&);

//-----------------------------------------------------------------------------
// Modifying the interface of BoundaryCondition
//-----------------------------------------------------------------------------
%rename (_function_space) dolfin::BoundaryCondition::function_space;

//-----------------------------------------------------------------------------
// Ignores domain assignment and operator== for Form class
//-----------------------------------------------------------------------------
%ignore dolfin::Form::dx;
%ignore dolfin::Form::ds;
%ignore dolfin::Form::dS;
%ignore dolfin::Form::operator==;

//-----------------------------------------------------------------------------
// Ignore dolfin::Cell versions of signatures as these now are handled by
// a typemap
//-----------------------------------------------------------------------------
%ignore dolfin::FiniteElement::evaluate_basis(uint i,
                                             double* values,
                                             const double* x,
                                             const Cell& cell) const;

%ignore dolfin::FiniteElement::evaluate_basis_all(double* values,
                                                  const double* coordinates,
                                                  const Cell& cell) const;

%ignore dolfin::DofMap::tabulate_coordinates(
                                    boost::multi_array<double, 2>& coordinates,
                                    const Cell& cell) const;

%ignore dolfin::GenericDofMap::tabulate_coordinates(
                                    boost::multi_array<double, 2>& coordinates,
                                    const Cell& cell) const;

%ignore dolfin::DofMap::tabulate_coordinates(
			      boost::multi_array<double, 2>& coordinates,
			      const ufc::cell& cell) const;

%ignore dolfin::GenericDofMap::tabulate_coordinates(
                              boost::multi_array<double, 2>& coordinates,
                              const ufc::cell& cell) const;

//-----------------------------------------------------------------------------
// Add a greedy typemap for dolfin::Cell to ufc::cell
//-----------------------------------------------------------------------------
%typemap(in) const ufc::cell& (void *argp)
{
  // const ufc::cell& cell Typemap
  int res = SWIG_ConvertPtr($input, &argp, $descriptor(dolfin::Cell*), 0);
  if (!SWIG_IsOK(res))
    SWIG_exception(SWIG_TypeError, "expected a dolfin.Cell");

  $1 = new dolfin::UFCCell(*reinterpret_cast<dolfin::Cell *>(argp));
}

%typemap(freearg) const ufc::cell&
{
  delete $1;
}

%typecheck(SWIG_TYPECHECK_POINTER) const ufc::cell& {
  int res = SWIG_ConvertPtr($input, 0, $descriptor(dolfin::Cell*), 0);
  $1 = SWIG_CheckState(res);
}

//-----------------------------------------------------------------------------
// Instantiate Hierarchical classes
//-----------------------------------------------------------------------------
namespace dolfin
{
  class Form;
  class LinearVariationalProblem;
  class NonlinearVariationalProblem;
  class DirichletBC;
}

%template (HierarchicalForm) dolfin::Hierarchical<dolfin::Form>;
%template (HierarchicalLinearVariationalProblem) \
          dolfin::Hierarchical<dolfin::LinearVariationalProblem>;
%template (HierarchicalNonlinearVariationalProblem) \
          dolfin::Hierarchical<dolfin::NonlinearVariationalProblem>;
%template (HierarchicalDirichletBC) dolfin::Hierarchical<dolfin::DirichletBC>;
