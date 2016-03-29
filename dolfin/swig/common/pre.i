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
// This must come early. The petsc4py/slepc4py module defines typemaps
// which we will later use on %extended classes (in post).  The
// typemaps must be in scope when swig sees the original class, not
// the extended definition.

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
// Avoid polluting the dolfin namespace with symbols introduced in
// ufc.h that we don't need
%rename("$ignore", regextarget=1, fullname=1) "ufc::.*$";

// Brings back ('un-ignore') some destructors for SWIG to handle
// shared_ptr wrapping
%rename("%s", regextarget=1, fullname=1) "ufc::cell::~cell()$";
%rename("%s", regextarget=1, fullname=1) "ufc::dofmap::~dofmap()$";
%rename("%s", regextarget=1, fullname=1) "ufc::finite_element::~finite_element()*$";
%rename("%s", regextarget=1, fullname=1) "ufc::form::~form()*$";
%rename("%s", regextarget=1, fullname=1) "ufc::function::~function()*$";

// Bring back ufc::cell members that are used in user Python
// implementations of Expression.eval_cell
%rename(cell_shape) ufc::cell::cell_shape();
%rename(index) ufc::cell::index;
%rename(topological_dimension) ufc::cell::topological_dimension;
%rename(geometric_dimension) ufc::cell::geometric_dimension;
%rename(local_facet) ufc::cell::local_facet;
%rename(mesh_identifier) ufc::cell::mesh_identifier;

// Rename only the symbols we need to ufc_* 'namespace'
%rename(ufc_cell) ufc::cell;
%rename(ufc_dofmap) ufc::dofmap;
%rename(ufc_finite_element) ufc::finite_element;
%rename(ufc_form) ufc::form;
%rename(ufc_function) ufc::function;
%include <ufc.h>

// Jit with ctypes will result in factory functions returning just a
// void * to a new object, these functions will cast them into our
// swig wrapper type system and make them shared_ptrs in the process
// to manage their lifetime.
%inline %{
std::shared_ptr<const ufc::finite_element> make_ufc_finite_element(void * element)
{
  ufc::finite_element * p = static_cast<ufc::finite_element *>(element);
  return std::shared_ptr<const ufc::finite_element>(p);
}

std::shared_ptr<const ufc::finite_element> make_ufc_finite_element(std::size_t element)
{
  ufc::finite_element * p = reinterpret_cast<ufc::finite_element *>(element);
  return std::shared_ptr<const ufc::finite_element>(p);
}

std::shared_ptr<const ufc::dofmap> make_ufc_dofmap(void * dofmap)
{
  ufc::dofmap * p = static_cast<ufc::dofmap *>(dofmap);
  return std::shared_ptr<const ufc::dofmap>(p);
}

std::shared_ptr<const ufc::dofmap> make_ufc_dofmap(std::size_t dofmap)
{
  ufc::dofmap * p = reinterpret_cast<ufc::dofmap *>(dofmap);
  return std::shared_ptr<const ufc::dofmap>(p);
}

std::shared_ptr<const ufc::form> make_ufc_form(void * form)
{
  ufc::form * p = static_cast<ufc::form *>(form);
  return std::shared_ptr<const ufc::form>(p);
}

std::shared_ptr<const ufc::form> make_ufc_form(std::size_t form)
{
  ufc::form * p = reinterpret_cast<ufc::form *>(form);
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
// Ignore all foo and rename foo_shared_ptr to _foo
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
