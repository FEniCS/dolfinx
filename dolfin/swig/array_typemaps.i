/* -*- C -*- */
// Copyright (C) 2009 Johan Hake
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-12-10
// Last changed: 2010-03-07

//=============================================================================
// In this file we declare some typemaps for the dolfin:Array type
//=============================================================================

//-----------------------------------------------------------------------------
// Director typemaps for dolfin::Array
//-----------------------------------------------------------------------------
%typemap(directorin) const dolfin::Array<double>& x {
  npy_intp dims[1] = {$1_name.size()};
  double * data = const_cast<double*>($1_name.data().get());
  $input = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, reinterpret_cast<char *>(data));
 }

%typemap(directorin) dolfin::Array<double>& values {
  npy_intp dims[1] = {$1_name.size()};
  $input = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE,
				     reinterpret_cast<char *>($1_name.data().get()));
 }

//-----------------------------------------------------------------------------
// Macro for defining an in-typemap for NumPy array -> dolfin::Array
//
// TYPE       : The pointer type
// TYPECHECK  : The SWIG specific name of the type used in the array type checks values
//              SWIG use: INT32 for integer, DOUBLE for double aso.
// NUMPYTYPE  : The NumPy type that is going to be checked for
// TYPENAME   : The name of the pointer type, 'double' for 'double', 'uint' for
//              'dolfin::uint'
// DESCR      : The char descriptor of the NumPy type
// ARGNAME    : The name of the argument the typemap will kick in for pass nothing
//              and the typemap will kick in for all argument names
// CONSTARRAY : If the dolfin::Array is of type const, then pass const for this
//              argument
//-----------------------------------------------------------------------------
%define NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(TYPE, TYPECHECK, NUMPYTYPE, TYPENAME, DESCR, ARGNAME, CONSTARRAY)

%typecheck(SWIG_TYPECHECK_ ## TYPECHECK ## _ARRAY) CONSTARRAY dolfin::Array<TYPE> & ARGNAME{
    $1 = PyArray_Check($input) ? 1 : 0;
}

%typemap(in) CONSTARRAY dolfin::Array<TYPE> &ARGNAME{
  // Check input object
  if (!PyArray_Check($input))
    SWIG_exception(SWIG_TypeError, "numpy array of 'TYPE_NAME' expected for the $argnum argument. Make sure that the numpy array use dtype='DESCR'.");

  PyArrayObject *xa = reinterpret_cast<PyArrayObject*>($input);
  if (PyArray_TYPE(xa) != NUMPYTYPE )
    SWIG_exception(SWIG_TypeError, "numpy array of 'TYPE_NAME' expected for the $argnum argument. Make sure that the numpy array use dtype='DESCR'.");

  dolfin::uint size = PyArray_DIM(xa, 0);
  TYPE* data = static_cast<TYPE*>(PyArray_DATA(xa));

  $1 = new dolfin::Array<TYPE>(size, data);
}

// We cannot create
%typemap(freearg) CONSTARRAY dolfin::Array<TYPE> & ARGNAME{
  delete $1;
}

%enddef

//-----------------------------------------------------------------------------
// Run the typemap macros
//-----------------------------------------------------------------------------

// Instantiate argument name specific typemaps for non const arguments
NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(double, DOUBLE, NPY_DOUBLE, double, d, values, )
NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(dolfin::uint, INT32, NPY_UINT, uint, I, indices,)

// Instantiate argument name independent typemaps for all
// const Array <{int, uint, double}>& arguments
NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(double, DOUBLE, NPY_DOUBLE, double, d, , const)
NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(dolfin::uint, INT32, NPY_UINT, uint, I, , const)
NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(int, INT32, NPY_INT, int, i, , const)


