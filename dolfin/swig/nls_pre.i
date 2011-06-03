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
// First added:  2009-10-07
// Last changed: 2011-01-19

//=============================================================================
// SWIG directives for the DOLFIN nls kernel module (pre)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
//=============================================================================

//-----------------------------------------------------------------------------
// Add director classes
//-----------------------------------------------------------------------------
%feature("director") dolfin::NonlinearProblem;

//-----------------------------------------------------------------------------
// Director typemaps for dolfin::GenericFoo
//-----------------------------------------------------------------------------
// FIXME: Why are not the typemaps defined by this macro kicking in?!?!
%define DIRECTORIN_TYPEMAPS(TYPE, CONST)

%typemap(directorin) CONST TYPE& {
  SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > *smartresult = new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >(reference_to_no_delete_pointer($1_name));
  $input = SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< TYPE > *), SWIG_POINTER_OWN);
}

%enddef

DIRECTORIN_TYPEMAPS(dofin::GenericMatrix, )
DIRECTORIN_TYPEMAPS(dofin::GenericVector, )
DIRECTORIN_TYPEMAPS(dofin::GenericVector, const)

%typemap(directorin) dolfin::GenericMatrix& {
    SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::GenericMatrix > *smartresult = new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::GenericMatrix >(reference_to_no_delete_pointer($1_name));
  $input = SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::GenericMatrix > *), SWIG_POINTER_OWN);
}

%typemap(directorin) dolfin::GenericVector& {
  SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::GenericVector > *smartresult = new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::GenericVector >(reference_to_no_delete_pointer($1_name));
  $input = SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::GenericVector > *), SWIG_POINTER_OWN);
}

%typemap(directorin) const dolfin::GenericVector& {
  SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< const dolfin::GenericVector > *smartresult = new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< const dolfin::GenericVector >(reference_to_no_delete_pointer($1_name));
  $input = SWIG_NewPointerObj(%as_voidptr(smartresult), $descriptor(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::GenericVector > *), SWIG_POINTER_OWN);
}
