/* -*- C -*- */
// Copyright (C) 2009 Johan Hake
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
// First added:  2009-09-22
// Last changed: 2014-01-17

//=============================================================================
// SWIG directives for the DOLFIN common kernel module (pre)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
//=============================================================================

//-----------------------------------------------------------------------------
// petsc4py/slepc4py typemaps
//-----------------------------------------------------------------------------
// This must come early. The petsc4py/slepc4py module defines typemaps which
// we will later use on %extended classes (in post).  The typemaps
// must be in scope when swig sees the original class, not the
// extended definition.

// Remove petsc4py typemaps that check for nullity of pointer
// and object itself we only care about the former.
%define %petsc4py_objreft(Type)
%typemap(check,noblock=1) Type *OUTPUT {
  if ($1 == NULL)
    %argument_nullref("$type", $symname, $argnum);
 }
%apply Type *OUTPUT { Type & }
%enddef

#ifdef HAS_PETSC4PY
%include "petsc4py/petsc4py.i"
%petsc4py_objreft(Mat)
%petsc4py_objreft(Vec)
%petsc4py_objreft(KSP)
%petsc4py_objreft(SNES)
%petsc4py_objreft(Tao)
#endif

#ifdef HAS_SLEPC4PY
%include "slepc4py/slepc4py.i"
%petsc4py_objreft(EPS)
#endif

//-----------------------------------------------------------------------------
// Make DOLFIN aware of the types defined in UFC
//-----------------------------------------------------------------------------
%{
#include <ufc.h>
%}
%ignore ufc::integral;
%ignore ufc::cell_integral;
%ignore ufc::exterior_facet_integral;
%ignore ufc::interior_facet_integral;
%ignore ufc::vertex_integral;
%ignore ufc::custom_integral;
%ignore ufc::cutcell_integral;
%ignore ufc::interface_integral;
%ignore ufc::overlap_integral;
%rename(ufc_function) ufc::function;
%include <ufc.h>

std::shared_ptr<const ufc::finite_element> make_ufc_finite_element(void * element);
std::shared_ptr<const ufc::dofmap> make_ufc_dofmap(void * dofmap);
std::shared_ptr<const ufc::form> make_ufc_form(void * dofmap);

%inline %{
std::shared_ptr<const ufc::finite_element> make_ufc_finite_element(void * element)
{
  ufc::finite_element * p = static_cast<ufc::finite_element *>(element);
  return std::shared_ptr<const ufc::finite_element>(p);
}

std::shared_ptr<const ufc::dofmap> make_ufc_dofmap(void * dofmap)
{
  ufc::dofmap * p = static_cast<ufc::dofmap *>(dofmap);
  return std::shared_ptr<const ufc::dofmap>(p);
}

std::shared_ptr<const ufc::form> make_ufc_form(void * form)
{
  ufc::form * p = static_cast<ufc::form *>(form);
  return std::shared_ptr<const ufc::form>(p);
}
%}

//-----------------------------------------------------------------------------
// Global modifications to the Array interface
//-----------------------------------------------------------------------------
%ignore dolfin::Array::operator=;
%ignore dolfin::Array::operator[];

//-----------------------------------------------------------------------------
// Global modifications to the ArrayView interface
//-----------------------------------------------------------------------------
%ignore dolfin::ArrayView::operator=;
%ignore dolfin::ArrayView::operator[];

//-----------------------------------------------------------------------------
// Global modifications to the IndexSet interface
//-----------------------------------------------------------------------------
%ignore dolfin::IndexSet::operator[];

//-----------------------------------------------------------------------------
// Global modifications to the dolfin::Set interface
//-----------------------------------------------------------------------------
%ignore dolfin::Set::operator[];

//-----------------------------------------------------------------------------
// Copy Array construction typemaps from NumPy typemaps
//-----------------------------------------------------------------------------
%typemap(in) (std::size_t N, const std::size_t* x) = (std::size_t _array_dim, std::size_t* _array);
%typemap(in) (std::size_t N, const int* x) = (std::size_t _array_dim, int* _array);
%typemap(in) (std::size_t N, const double* x) = (std::size_t _array_dim, double* _array);

//%typemap(in) (std::size_t N, const std::size_t* x) = (std::size_t _arrayview_dim, std::size_t* _arrayview);
//%typemap(in) (std::size_t N, const int* x) = (std::size_t _arrayview_dim, int* _arrayview);
//%typemap(in) (std::size_t N, const double* x) = (std::size_t _arrayview_dim, double* _arrayview);

//-----------------------------------------------------------------------------
// Ignores for Hierarchical
//-----------------------------------------------------------------------------
%ignore dolfin::Hierarchical::operator=;

//-----------------------------------------------------------------------------
// Ignore all foo and rename foo_shared_ptr to _foo for SWIG >= 2.0
// and ignore foo_shared_ptr for SWIG < 2.0
//-----------------------------------------------------------------------------
%ignore dolfin::Hierarchical::parent;
%rename(_parent) dolfin::Hierarchical::parent_shared_ptr;
%ignore dolfin::Hierarchical::child;
%rename(_child) dolfin::Hierarchical::child_shared_ptr;
%ignore dolfin::Hierarchical::root_node;
%rename(_root_node) dolfin::Hierarchical::root_node_shared_ptr;
%ignore dolfin::Hierarchical::leaf_node;
%rename(_leaf_node) dolfin::Hierarchical::leaf_node_shared_ptr;

//-----------------------------------------------------------------------------
// Ignores for Variable
//-----------------------------------------------------------------------------
%ignore dolfin::Variable::operator=;
