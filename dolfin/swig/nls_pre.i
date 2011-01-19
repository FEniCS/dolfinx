/* -*- C -*- */
// Copyright (C) 2009 Johan Hake
// Licensed under the GNU LGPL Version 2.1.
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
