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
// Modified by Garth Wells, 2007-2012.
// Modified by Kent-Andre Mardal, 2008.
// Modified by Johan Hake, 2008-2009.
//
// First added:  2007-08-16
// Last changed: 2012-02-29

// ===========================================================================
// SWIG directives for the DOLFIN fem kernel module (pre)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
// ===========================================================================

//-----------------------------------------------------------------------------
// Rename solve so it wont clash with solve from la
//-----------------------------------------------------------------------------
%rename(fem_solve) dolfin::solve;

//-----------------------------------------------------------------------------
// Modifying the interface of FooProblem
//-----------------------------------------------------------------------------
%define PROBLEM_RENAMES(NAME)
%rename(_solution) dolfin::NAME ## Problem::solution;
%rename(_trial_space) dolfin::NAME ## Problem::trial_space;
%rename(_test_space) dolfin::NAME ## Problem::test_space;
%enddef

PROBLEM_RENAMES(LinearVariational)
PROBLEM_RENAMES(NonlinearVariational)
//PROBLEM_RENAMES(LinearTimeDependent)

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
					 const MeshFunction<std::size_t>&,
					 std::size_t,
					 std::string method="topological");
%ignore dolfin::DirichletBC::DirichletBC(const FunctionSpace&,
					 const GenericFunction&,
					 std::size_t,
					 std::string method="topological");
%ignore dolfin::DirichletBC::DirichletBC(boost::shared_ptr<const FunctionSpace>,
					 boost::shared_ptr<const GenericFunction>,
					 const std::vector<std::pair<std::size_t, std::size_t> >&,
					 std::string method="topological");

%ignore dolfin::LinearVariationalProblem::LinearVariationalProblem(const Form&,
                                                                   const Form&,
                                                                   Function&);
%ignore dolfin::LinearVariationalProblem::LinearVariationalProblem(const Form&,
                                                     const Form&,
                                                     Function&,
                                                     const DirichletBC&);
%ignore dolfin::LinearVariationalProblem::LinearVariationalProblem(const Form&,
                                       const Form&,
                                       Function&,
                                       std::vector<const DirichletBC*>);

%ignore dolfin::NonlinearVariationalProblem::NonlinearVariationalProblem(const Form&,
                                                                     Function&);
%ignore dolfin::NonlinearVariationalProblem::NonlinearVariationalProblem(const Form&,
                                                                   Function&,
                                                                   const Form&);
%ignore dolfin::NonlinearVariationalProblem::NonlinearVariationalProblem(const Form&,
                                                     Function&,
                                                     const DirichletBC&);
%ignore dolfin::NonlinearVariationalProblem::NonlinearVariationalProblem(const Form&,
                                                     Function&,
                                                     const DirichletBC&,
                                                     const Form&);
%ignore dolfin::NonlinearVariationalProblem::NonlinearVariationalProblem(const Form&,
                                       Function&,
                                       std::vector<const DirichletBC*>);
%ignore dolfin::NonlinearVariationalProblem::NonlinearVariationalProblem(const Form&,
                                         Function&,
                                         std::vector<const DirichletBC*>,
                                         const Form&);

%ignore dolfin::LinearVariationalSolver::LinearVariationalSolver(LinearVariationalProblem&);

%ignore dolfin::NonlinearVariationalSolver::NonlinearVariationalSolver(NonlinearVariationalProblem&);

%ignore dolfin::SystemAssembler(const Form& a, const Form&);
%ignore dolfin::SystemAssembler(const Form& a, const Form&, const DirichletBC&);
%ignore dolfin::SystemAssembler(const Form& a, const Form&,
                                const std::vector<const DirichletBC*>);

//-----------------------------------------------------------------------------
// Ignore operator= for DirichletBC to avoid warning
//-----------------------------------------------------------------------------
%ignore dolfin::DirichletBC::operator=;

//-----------------------------------------------------------------------------
// Modifying the interface of DirichletBC
//-----------------------------------------------------------------------------
%rename (_function_space) dolfin::DirichletBC::function_space;

//-----------------------------------------------------------------------------
// Modifying the interface of Form
//-----------------------------------------------------------------------------
%rename (_function_space) dolfin::Form::function_space;

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
%ignore dolfin::FiniteElement::evaluate_basis(std::size_t i,
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
%typemap(in) const ufc::cell& (void *argp, bool dolfin_cell, int res)
{
  // const ufc::cell& cell Typemap
  // First try dolfin::Cell
  res = SWIG_ConvertPtr($input, &argp, $descriptor(dolfin::Cell*), 0);
  if (SWIG_IsOK(res))
  {
    dolfin_cell = true;
    $1 = new dolfin::UFCCell(*reinterpret_cast<dolfin::Cell *>(argp));
  }

  else
  {
    dolfin_cell = false;
    res = SWIG_ConvertPtr($input, &argp, $descriptor(ufc::cell*), 0);
    if (SWIG_IsOK(res))
      $1 = reinterpret_cast<ufc::cell *>(argp);
    else
      SWIG_exception(SWIG_TypeError, "expected a dolfin.Cell or a ufc::cell");
  }
}

%typemap(freearg) const ufc::cell&
{
  // If a dolfin cell was created delete it
  if(dolfin_cell$argnum)
    delete $1;
}

%typecheck(SWIG_TYPECHECK_POINTER) const ufc::cell&
{
  // TYPECHECK const ufc::cell&
  int res = SWIG_ConvertPtr($input, 0, $descriptor(dolfin::Cell*), 0);
  $1 = SWIG_CheckState(res);
  if (!$1)
  {
    res = SWIG_ConvertPtr($input, 0, $descriptor(ufc::cell*), 0);
    $1 = SWIG_CheckState(res);
  }
}

%define IN_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_SHARED_POINTERS(TYPE)

//-----------------------------------------------------------------------------
// The std::vector<std::vector<SWIG_SHARED_PTR_QNAMESPACE::shared_ptr<dolfin::Type*> > > 
// typecheck
//-----------------------------------------------------------------------------
%typecheck(SWIG_TYPECHECK_POINTER) std::vector<std::vector<SWIG_SHARED_PTR_QNAMESPACE::shared_ptr<const dolfin::TYPE> > >
{
  $1 = PyList_Check($input) ? 1 : 0;
}

//-----------------------------------------------------------------------------
// The std::vector<std::vector<SWIG_SHARED_PTR_QNAMESPACE::shared_ptr<dolfin::Type*> > > 
// typemap
//-----------------------------------------------------------------------------
   %typemap (in) std::vector<std::vector<SWIG_SHARED_PTR_QNAMESPACE::shared_ptr<const dolfin::TYPE> > > (std::vector<std::vector<SWIG_SHARED_PTR_QNAMESPACE::shared_ptr<const dolfin::TYPE> > >  tmp_vec_0, std::vector<SWIG_SHARED_PTR_QNAMESPACE::shared_ptr<const dolfin::TYPE> >  tmp_vec_1, SWIG_SHARED_PTR_QNAMESPACE::shared_ptr<dolfin::TYPE> tempshared)
{
  // IN_TYPEMAP_STD_VECTOR_OF_POINTERS(TYPE, CONST, CONST_VECTOR), shared_ptr version
  if (!PyList_Check($input))
    SWIG_exception(SWIG_TypeError, "list of lists of TYPE expected");

  // Size of first list
  int size_0 = PyList_Size($input);
  int res = 0;
  PyObject* py_item_0 = 0;
  PyObject* py_item_1 = 0;
  void* itemp = 0;
  int newmem = 0;
  tmp_vec_0.reserve(size_0);

  // Iterate over first list
  for (int i = 0; i < size_0; i++)
  {
    py_item_0 = PyList_GetItem($input, i);

    // Check list items are list
    if (!PyList_Check(py_item_0))
      SWIG_exception(SWIG_TypeError, "list of lists of TYPE expected");
    
    // Size of second list
    int size_1 = PyList_Size(py_item_0);
    tmp_vec_1.reserve(size_1);

    // Iterate over second list
    for (int j = 0; j < size_1; j++)
    {
      newmem = 0;

      // Grab item from second list
      py_item_1 = PyList_GetItem(py_item_0, j);

      // Try convert it
      res = SWIG_ConvertPtrAndOwn(py_item_1, &itemp, $descriptor(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::TYPE > *), 0, &newmem);
      
      if (!SWIG_IsOK(res))
	SWIG_exception(SWIG_TypeError, "expected a list of list of TYPE (Bad conversion)");  

      tempshared = *(reinterpret_cast<SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::TYPE> *>(itemp));
      tmp_vec_1.push_back(tempshared);

      if (newmem & SWIG_CAST_NEW_MEMORY)
	delete reinterpret_cast<SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::TYPE> *>(itemp);
    }
    tmp_vec_0.push_back(tmp_vec_1);
    tmp_vec_1.clear();
  }
  $1 = tmp_vec_0;

}

%enddef

//-----------------------------------------------------------------------------
// Instantiate typemap
//-----------------------------------------------------------------------------
IN_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_SHARED_POINTERS(Form)

//-----------------------------------------------------------------------------
// Instantiate Hierarchical classes
//-----------------------------------------------------------------------------
#ifdef FEMMODULE // Conditional template instiantiation for FEM module
%template (HierarchicalForm) dolfin::Hierarchical<dolfin::Form>;
%template (HierarchicalLinearVariationalProblem) \
          dolfin::Hierarchical<dolfin::LinearVariationalProblem>;
%template (HierarchicalNonlinearVariationalProblem) \
          dolfin::Hierarchical<dolfin::NonlinearVariationalProblem>;
%template (HierarchicalDirichletBC) dolfin::Hierarchical<dolfin::DirichletBC>;

#endif
//#ifdef IOMODULE // Conditional template instiantiation for IO module
//%template (HierarchicalDirichletBC) dolfin::Hierarchical<dolfin::DirichletBC>;
//#endif

