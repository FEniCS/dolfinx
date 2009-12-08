/* -*- C -*- */
// Copyright (C) 2009 Johan Hake
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-09-22
// Last changed: 2009-09-23

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

