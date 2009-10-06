/* -*- C -*- */
// Copyright (C) 2007-2009 Anders Logg
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Hake, 2008-2009.
//
// First added:  2007-04-17
// Last changed: 2009-10-06

//-----------------------------------------------------------------------------
// SWIG directives for directors for PyDOLFIN
//
// Directors facilitates python callbacks to the C++ library
//-----------------------------------------------------------------------------
%feature("director") Expression;
%feature("director") SubDomain;
%feature("director") NonlinearProblem;
%feature("director") ODE;
%feature("director") PETScKrylovMatrix;
%feature("director") uBLASKrylovMatrix;

//-----------------------------------------------------------------------------
// Typemap for values (in Expression)
//-----------------------------------------------------------------------------
%typemap(directorin) double* values {
  {
    // Compute size of value (number of entries in tensor value)
    dolfin::uint size = 1;
    for (dolfin::uint i = 0; i < this->value_rank(); i++)
      size *= this->value_dimension(i);

    npy_intp dims[1] = {size};
    $input = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, reinterpret_cast<char *>($1_name));
  }
}

//-----------------------------------------------------------------------------
// Typemap for coordinates (in Expression and SubDomain)
//-----------------------------------------------------------------------------
%typemap(directorin) const double* x {
  {
    // Compute size of x
    npy_intp dims[1] = {this->geometric_dimension()};
    $input = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, reinterpret_cast<char *>(const_cast<double*>($1_name)));
  }
}

%typemap(directorin) double* y {
  {
    // Compute size of y
    npy_intp dims[1] = {this->geometric_dimension()};
    $input = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, reinterpret_cast<char *>($1_name));
  }
}

