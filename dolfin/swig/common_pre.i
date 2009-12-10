/* -*- C -*- */
// Copyright (C) 2009 Johan Hake
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-09-22
// Last changed: 2009-12-08

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

%ignore dolfin::Array<const double>::Array(uint N);
%ignore dolfin::Array<const double>::resize(uint N);
%ignore dolfin::Array<const double>::zero();

%ignore dolfin::Array::operator=;
%ignore dolfin::Array::operator[];
%ignore dolfin::Array::Array(uint N, boost::shared_array<double> x);

/*
%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) (dolfin::uint N, double* x){
    $1 = PyArray_Check($input) ? 1 : 0;
}

// Typemap for Array constructor
%typemap(in) (dolfin::uint N, double* x){

  // Check input object
  if (!PyArray_Check($input))
    SWIG_exception(SWIG_TypeError, "NumPy array expected");
    
  PyArrayObject *xa = reinterpret_cast<PyArrayObject*>($input);
  if (PyArray_TYPE(xa) != NPY_DOUBLE )
    SWIG_exception(SWIG_TypeError, "NumPy array expected");
    
  $1 = PyArray_DIM(xa, 0);
  $2 = static_cast<double*>(PyArray_DATA(xa));
}
*/
