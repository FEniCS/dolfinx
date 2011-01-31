/* -*- C -*- */
// Copyright (C) 2009 Johan Hake
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-09-22
// Last changed: 2011-01-31

//=============================================================================
// SWIG directives for the DOLFIN real kernel module (pre)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
//=============================================================================

//-----------------------------------------------------------------------------
// Forward declare Parameters
//-----------------------------------------------------------------------------
namespace dolfin
{
  class Parameters;
}

//-----------------------------------------------------------------------------
// Ignore all functions using real*
// We need to hand an explicit typemap for this type
//-----------------------------------------------------------------------------
%ignore real_set(uint n, real* x, const real* y);
%ignore real_set(uint n, real* x, const real& value);
%ignore real_zero(uint n, real* x);
%ignore real_add(uint n, real* x, const real* y);
%ignore real_sub(uint n, real* x, const real* y);
%ignore real_axpy(uint n, real* x, const real& a, const real* y);
%ignore real_mult(uint n, real* x, const real& a);
%ignore real_div(uint n, real* x, const real& a);
%ignore real_inner(uint n, const real* x, const real* y);
%ignore real_max_abs(uint n, const real* x);
%ignore real_norm(uint n, const real* x);
%ignore real_identity(uint n, real* A, real value=1.0);

//-----------------------------------------------------------------------------
// Global modifications to the Array interface
//-----------------------------------------------------------------------------
%ignore dolfin::Array::operator=;
%ignore dolfin::Array::operator[];

//-----------------------------------------------------------------------------
// Global modifications to the dolfin::Set interface
//-----------------------------------------------------------------------------
%ignore dolfin::Set::operator[];

//-----------------------------------------------------------------------------
// Macro for defining an in-typemap for NumPy array -> dolfin::Array for the
// Array constructor
//
// TYPE       : The pointer type
// TYPECHECK  : The SWIG specific name of the type used in the array type checks values
//              SWIG use: INT32 for integer, DOUBLE for double aso.
// NUMPYTYPE  : The NumPy type that is going to be checked for
// TYPENAME   : The name of the pointer type, 'double' for 'double', 'uint' for
//              'dolfin::uint'
// DESCR      : The char descriptor of the NumPy type
//-----------------------------------------------------------------------------
%define ARRAY_CONSTRUCTOR_TYPEMAP(TYPE, TYPECHECK, NUMPYTYPE, TYPENAME, DESCR)

%typecheck(SWIG_TYPECHECK_ ## TYPECHECK ## _ARRAY) (dolfin::uint N, TYPE* x){
    $1 = PyArray_Check($input) ? 1 : 0;
}

%typemap(in) (dolfin::uint N, TYPE* x){

  // Check input object
  if (!PyArray_Check($input))
    SWIG_exception(SWIG_TypeError, "numpy array of 'TYPENAME' expected. Make sure that the numpy array use dtype='DESCR'.");

  PyArrayObject *xa = reinterpret_cast<PyArrayObject*>($input);
  if (PyArray_TYPE(xa) != NUMPYTYPE )
    SWIG_exception(SWIG_TypeError, "numpy array of 'TYPENAME' expected. Make sure that the numpy array use dtype='DESCR'.");

  $1 = PyArray_DIM(xa, 0);
  $2 = static_cast<TYPE*>(PyArray_DATA(xa));
}

%enddef

//-----------------------------------------------------------------------------
// Run Array typemap macros
//-----------------------------------------------------------------------------
ARRAY_CONSTRUCTOR_TYPEMAP(double, DOUBLE, NPY_DOUBLE, double, d)
// We nust use unsigned int here and not dolfin::uint, don't know why
ARRAY_CONSTRUCTOR_TYPEMAP(unsigned int, INT32, NPY_UINT, uint, I)
ARRAY_CONSTRUCTOR_TYPEMAP(int, INT32, NPY_INT, int, i)

//-----------------------------------------------------------------------------
// Ignores for Hierarchical
//-----------------------------------------------------------------------------
%ignore dolfin::Hierarchical::operator=;

//-----------------------------------------------------------------------------
// Forward declare Hierarchical template class
//-----------------------------------------------------------------------------
namespace dolfin {
  template<class T>
    class Hierarchical;
}

