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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2009-12-10
// Last changed: 2010-08-11

//=============================================================================
// In this file we declare some typemaps for the dolfin::Array type
//=============================================================================

//-----------------------------------------------------------------------------
// Macro for defining an in-typemap for NumPy array -> dolfin::Array
//
// TYPE       : The primitive type
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
%define IN_NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(TYPE, TYPECHECK, NUMPYTYPE, TYPENAME, DESCR, ARGNAME, CONSTARRAY)

%typemap(in) CONSTARRAY dolfin::Array<TYPE> &ARGNAME{
  // Check input object
  if (!PyArray_Check($input))
    SWIG_exception(SWIG_TypeError, "numpy array of 'TYPENAME' expected for the $argnum argument. Make sure that the numpy array use dtype='DESCR'.");

  PyArrayObject *xa = reinterpret_cast<PyArrayObject*>($input);
  if (PyArray_TYPE(xa) != NUMPYTYPE )
    SWIG_exception(SWIG_TypeError, "numpy array of 'TYPENAME' expected for the $argnum argument. Make sure that the numpy array use dtype='DESCR'.");

  dolfin::uint size = PyArray_DIM(xa, 0);
  TYPE* data = static_cast<TYPE*>(PyArray_DATA(xa));

  $1 = new dolfin::Array<TYPE>(size, data);
}

// Clean up the constructed Array
%typemap(freearg) CONSTARRAY dolfin::Array<TYPE> & ARGNAME{
  delete $1;
}

%typecheck(SWIG_TYPECHECK_ ## TYPECHECK ## _ARRAY) CONSTARRAY dolfin::Array<TYPE> & ARGNAME{
    $1 = PyArray_Check($input) ? 1 : 0;
}

%enddef

//-----------------------------------------------------------------------------
// Macro for defining an out-typemap for dolfin::Array -> NumPy array
//
// TYPE       : The primitive type
// NUMPYTYPE  : The NumPy type that is going to be checked for
//-----------------------------------------------------------------------------
%define OUT_NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(TYPE, NUMPYTYPE)

%typemap(out) dolfin::Array<TYPE> {

  // Create a swig wrapped Array which are going to be attached to the NumPy array
  PyObject *SWIG_array = 0;
  SWIG_array = SWIG_NewPointerObj(SWIG_as_voidptr(new dolfin::Array< TYPE >(*(&$1))), $descriptor(dolfin::Array< TYPE >*), SWIG_POINTER_OWN |  0 );

  // Create NumPy array
  npy_intp dims[1];
  dims[0] = (&$1)->size();
  PyArrayObject* numpy_array = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNewFromData(1, dims, NUMPYTYPE, (char *)((&$1)->data().get())));

  // Attach SWIG wrapped array to the numpy_array
  numpy_array->base = SWIG_array;
  
  if ( numpy_array == NULL ) 
    SWIG_exception(SWIG_TypeError, "Error in conversion of dolfin::Array< TYPE > to NumPy array.");

  // Return the NumPy array
  $result = reinterpret_cast<PyObject*>(numpy_array);
}
%enddef

//-----------------------------------------------------------------------------
// Director typemaps for dolfin::Array
//-----------------------------------------------------------------------------
%typemap(directorin) const dolfin::Array<double>& {
  npy_intp dims[1] = {$1_name.size()};
  double * data = const_cast<double*>($1_name.data().get());
  $input = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, reinterpret_cast<char *>(data));
 }

%typemap(directorin) dolfin::Array<double>& {
  npy_intp dims[1] = {$1_name.size()};
  $input = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE,
				     reinterpret_cast<char *>($1_name.data().get()));
 }

//-----------------------------------------------------------------------------
// Run the typemap macros
//-----------------------------------------------------------------------------

// Instantiate argument name specific typemaps for non const arguments
IN_NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(double, DOUBLE, NPY_DOUBLE, double, d, values, )
IN_NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(dolfin::uint, INT32, NPY_UINT, uint, I, indices,)
IN_NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(double, DOUBLE, NPY_DOUBLE, double, d, vertex_values, )

// Instantiate argument name independent typemaps for all
// const Array <{int, uint, double}>& arguments
IN_NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(double, DOUBLE, NPY_DOUBLE, double, d, , const)
IN_NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(dolfin::uint, INT32, NPY_UINT, uint, I, , const)
IN_NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(int, INT32, NPY_INT, int, i, , const)


OUT_NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(dolfin::uint, NPY_UINT)
OUT_NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(int, NPY_INT)
OUT_NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(double, NPY_DOUBLE)
