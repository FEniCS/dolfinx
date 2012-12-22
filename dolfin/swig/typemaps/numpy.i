/* -*- C -*- */
// Copyright (C) 2007-2009 Ola Skavhaug
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
// Modified by Johan Hake, 2008-2009.
// Modified by Anders logg, 2009.
//
// First added:  2007-12-16
// Last changed: 2011-04-29

//-----------------------------------------------------------------------------
// Function to set the base of an NumPy array object
//-----------------------------------------------------------------------------
%inline%{
PyObject* _attach_base_to_numpy_array(PyObject* obj, PyObject* owner)
{
  if (owner == NULL)
  {
    PyErr_SetString(PyExc_TypeError, "Expected a Python object as owner argument");
    return NULL;
  }

  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(obj);
  if (array == NULL)
  {
    PyErr_SetString(PyExc_TypeError, "NumPy conversion error");
    return NULL;
  }

  // Bump the references
  Py_XINCREF(owner);
  Py_XINCREF(Py_None);

  // Assign the base
  PyArray_BASE(array) = owner;

  return Py_None;
}
%}

//-----------------------------------------------------------------------------
// Help defines for using the generated numpy array wrappers
//-----------------------------------------------------------------------------
#define %make_numpy_array(dim, type_name) make_ ## dim ## d_numpy_array_ ## type_name
#define make_numpy_array_frag(dim, type_name) "make_" %str(dim) "d_numpy_array_" %str(type_name)

//-----------------------------------------------------------------------------
// A fragment function which takes a PyArray as a PyObject, check conversion and
// set the writable flags
//-----------------------------------------------------------------------------
%fragment("return_py_array", "header") {
SWIGINTERNINLINE PyObject* return_py_array(PyObject* obj, bool writable)
{
  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(obj);
  if ( obj == NULL )
  {
    PyErr_SetString(PyExc_TypeError, "NumPy conversion error");
    return NULL;
  }

  // Set writable flag on numpy array
  if ( !writable )
    array->flags &= ~NPY_WRITEABLE;
  return reinterpret_cast<PyObject*>(array);
}
}

//-----------------------------------------------------------------------------
// Macro for generating fragments to constructing NumPys array from data ponters
// The macro generates two functions, one for 1D and one for 2D arrays
//
// TYPE       : The pointer type
// NUMPY_TYPE : The NumPy type that is going to be checked for
// TYPE_NAME  : The name of the pointer type, 'double' for 'double', 'uint' for
//              'dolfin::uint', size_t for std::size_t
//
// Note that each invocation of this macro two functions will be inlined in
// the SWIG layer:
//
//    1) make_1d_numpy_array_{TYPE_NAME}
//    2) make_2d_numpy_array_{TYPE_NAME}
//
// Here TYPE_NAME is used to name the generated C++ function.
//-----------------------------------------------------------------------------
%define NUMPY_ARRAY_FRAGMENTS(TYPE, NUMPY_TYPE, TYPE_NAME)
%fragment(make_numpy_array_frag(2, TYPE_NAME), "header",
	  fragment="return_py_array") {
SWIGINTERNINLINE PyObject* %make_numpy_array(2, TYPE_NAME)
  (int m, int n, const TYPE* dataptr, bool writable = true)
{
  npy_intp adims[2] = {m, n};
  return return_py_array(PyArray_SimpleNewFromData(2, adims, NUMPY_TYPE,
						   (char *)(dataptr)), writable);
}}

%fragment(make_numpy_array_frag(1, TYPE_NAME), "header",
	  fragment="return_py_array") {
SWIGINTERNINLINE PyObject* %make_numpy_array(1, TYPE_NAME)
  (int m, const TYPE* dataptr, bool writable = true)
{
  npy_intp adims[1] = {m};
  return return_py_array(PyArray_SimpleNewFromData(1, adims, NUMPY_TYPE,
						   (char *)(dataptr)), writable);
}}

// Force the fragments to be instantiated
%fragment(make_numpy_array_frag(1, TYPE_NAME));
%fragment(make_numpy_array_frag(2, TYPE_NAME));

%enddef

//-----------------------------------------------------------------------------
// Macro for defining an unsafe in-typemap for NumPy arrays -> c arrays
//
// The typmaps defined by this macro just passes the pointer to the C array,
// contained in the NumPy array to the function. The knowledge of the length
// of the incomming array is not used.
//
// TYPE       : The pointer type
// TYPE_UPPER : The SWIG specific name of the type used in the array type checks values
//              SWIG use: INT32 for integer, DOUBLE for double aso.
// NUMPY_TYPE : The NumPy type that is going to be checked for
// TYPE_NAME  : The name of the pointer type, 'double' for 'double', 'uint' for
//              'dolfin::uint'
// DESCR      : The char descriptor of the NumPy type
//-----------------------------------------------------------------------------

#define convert_numpy_to_array_no_check(Type) "convert_numpy_to_array_no_check_" {Type}

%define UNSAFE_NUMPY_TYPEMAPS(TYPE,TYPE_UPPER,NUMPY_TYPE,TYPE_NAME,DESCR)

%fragment(convert_numpy_to_array_no_check(TYPE_NAME), "header") {
//-----------------------------------------------------------------------------
// Typemap function (Reducing wrapper code size)
//-----------------------------------------------------------------------------
SWIGINTERN bool convert_numpy_to_array_no_check_ ## TYPE_NAME(PyObject* input, TYPE*& ret)
{
  if (PyArray_Check(input))
  {
    PyArrayObject *xa = reinterpret_cast<PyArrayObject*>(input);
    if (PyArray_ISCONTIGUOUS(xa) && PyArray_TYPE(xa) == NUMPY_TYPE)
    {
      ret  = static_cast<TYPE*>(PyArray_DATA(xa));
      return true;
    }
  }
  PyErr_SetString(PyExc_TypeError,"contiguous numpy array of 'TYPE_NAME' expected. Make sure that the numpy array is contiguous, and uses dtype=DESCR.");
  return false;
}
}

//-----------------------------------------------------------------------------
// The typecheck
//-----------------------------------------------------------------------------
%typecheck(SWIG_TYPECHECK_ ## TYPE_UPPER ## _ARRAY) TYPE *
{
    $1 = PyArray_Check($input) ? 1 : 0;
}

//-----------------------------------------------------------------------------
// The typemap
//-----------------------------------------------------------------------------
%typemap(in, fragment=convert_numpy_to_array_no_check(TYPE_NAME)) TYPE *
{
if (!convert_numpy_to_array_no_check_ ## TYPE_NAME($input,$1))
    return NULL;
}

//-----------------------------------------------------------------------------
// Apply the typemap on the TYPE* _array argument
//-----------------------------------------------------------------------------
%apply TYPE* {TYPE* _array}

%enddef

//-----------------------------------------------------------------------------
// Macro for defining an safe in-typemap for NumPy arrays -> c arrays
//
// Type       : The pointer type
// TYPE_UPPER : The SWIG specific name of the type used in the array type checks values
//              SWIG use: INT32 for integer, DOUBLE for double aso.
// NUMPY_TYPE : The NumPy type that is going to be checked for
// TYPE_NAME  : The name of the pointer type, 'double' for 'double', 'uint' for
//              'dolfin::uint'
// DESCR      : The char descriptor of the NumPy type
//-----------------------------------------------------------------------------
#define convert_numpy_to_array_with_check(Type) "convert_numpy_to_array_with_check_" {Type}
%define SAFE_NUMPY_TYPEMAPS(TYPE,TYPE_UPPER,NUMPY_TYPE,TYPE_NAME,DESCR)

%fragment(convert_numpy_to_array_with_check(TYPE_NAME), "header") {
//-----------------------------------------------------------------------------
// Typemap function (Reducing wrapper code size)
//-----------------------------------------------------------------------------
SWIGINTERN bool convert_numpy_to_array_with_check_ ## TYPE_NAME(PyObject* input, std::size_t& _array_dim, TYPE*& _array)
{
  if (PyArray_Check(input))
  {
    PyArrayObject *xa = reinterpret_cast<PyArrayObject*>(input);
    if (PyArray_ISCONTIGUOUS(xa) && (PyArray_TYPE(xa) == NUMPY_TYPE) &&
          (PyArray_NDIM(xa)==1))
    {
      _array  = static_cast<TYPE*>(PyArray_DATA(xa));
      _array_dim = static_cast<unsigned int>(PyArray_DIM(xa,0));
      return true;
    }
  }
  PyErr_SetString(PyExc_TypeError,"contiguous numpy array of 'TYPE_NAME' expected. Make sure that the numpy array is contiguous, with 1 dimension, and uses dtype=DESCR.");
  return false;
}
}

//-----------------------------------------------------------------------------
// The typecheck
//-----------------------------------------------------------------------------
%typecheck(SWIG_TYPECHECK_ ## TYPE_UPPER ## _ARRAY) (std::size_t _array_dim, TYPE* _array)
{
  $1 = PyArray_Check($input) ? 1 : 0;
}

//-----------------------------------------------------------------------------
// The typemap
//-----------------------------------------------------------------------------
%typemap(in, fragment=convert_numpy_to_array_with_check(TYPE_NAME)) \
  (std::size_t _array_dim, TYPE* _array)
{
  if (!convert_numpy_to_array_with_check_ ## TYPE_NAME($input,$1,$2))
    return NULL;
}
%enddef

//-----------------------------------------------------------------------------
// Run the different macros and instantiate the typemaps
// NOTE: If a typemap is not used an error will be issued as the generated
//       typemap function will not be used
//-----------------------------------------------------------------------------
UNSAFE_NUMPY_TYPEMAPS(std::size_t, INT32, NPY_UINTP, size_t, uintp)
UNSAFE_NUMPY_TYPEMAPS(double,DOUBLE,NPY_DOUBLE,double,float_)
UNSAFE_NUMPY_TYPEMAPS(dolfin::DolfinIndex,INT32,NPY_UINT,dolfin_index,intc)
//UNSAFE_NUMPY_TYPEMAPS(int,INT,NPY_INT,int,cint)

SAFE_NUMPY_TYPEMAPS(std::size_t,INT32,NPY_UINTP,size_t,uintp)
SAFE_NUMPY_TYPEMAPS(dolfin::DolfinIndex,INT32,NPY_INT,dolfin_index,intc)
SAFE_NUMPY_TYPEMAPS(double,DOUBLE,NPY_DOUBLE,double,float_)
SAFE_NUMPY_TYPEMAPS(int,INT32,NPY_INT,int,cint)

// Instantiate the code used by the make_numpy_array macro.
// The first argument name the C++ type, the second the corresponding
// NumPy type and the third argument a shorthand name for the C++ type
// to identify the correct function
NUMPY_ARRAY_FRAGMENTS(unsigned int, NPY_UINT, uint)
NUMPY_ARRAY_FRAGMENTS(double, NPY_DOUBLE, double)
NUMPY_ARRAY_FRAGMENTS(int, NPY_INT, int)
NUMPY_ARRAY_FRAGMENTS(bool, NPY_BOOL, bool)
NUMPY_ARRAY_FRAGMENTS(std::size_t, NPY_UINTP, size_t)
NUMPY_ARRAY_FRAGMENTS(dolfin::DolfinIndex, NPY_INT, dolfin_index)

//-----------------------------------------------------------------------------
// Typecheck for function expecting two-dimensional NumPy arrays of double
//-----------------------------------------------------------------------------
%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) (int _array_dim_0, int _array_dim_1, double* _array)
{
    $1 = PyArray_Check($input) ? 1 : 0;
}

//-----------------------------------------------------------------------------
// Generic typemap to expand a two-dimensional NumPy arrays into three
// C++ arguments: _array_dim_0, _array_dim_1, _array
//-----------------------------------------------------------------------------
%typemap(in) (int _array_dim_0, int _array_dim_1, double* _array)
{
  if (PyArray_Check($input))
  {
    PyArrayObject *xa = reinterpret_cast<PyArrayObject*>($input);
    if ( PyArray_TYPE(xa) == NPY_DOUBLE )
    {
      if ( PyArray_NDIM(xa) == 2 )
      {
        $1 = PyArray_DIM(xa,0);
        $2 = PyArray_DIM(xa,1);
        $3  = static_cast<double*>(PyArray_DATA(xa));
      }
      else
      {
        SWIG_exception(SWIG_ValueError, "2d Array expected");
      }
    }
    else
    {
      SWIG_exception(SWIG_TypeError, "Array of doubles expected");
    }
  }
  else
  {
    SWIG_exception(SWIG_TypeError, "Array expected");
  }
}

//-----------------------------------------------------------------------------
// Typecheck for function expecting two-dimensional NumPy arrays of int
//-----------------------------------------------------------------------------
%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) (int _array_dim_0, int _array_dim_1, int* _array)
{
    $1 = PyArray_Check($input) ? 1 : 0;
}

//-----------------------------------------------------------------------------
// Generic typemap to expand a two-dimensional NumPy arrays into three
// C++ arguments: _array_dim_0, _array_dim_1, _array
//-----------------------------------------------------------------------------
%typemap(in) (int _array_dim_0, int _array_dim_1, int* _array)
{
    if PyArray_Check($input) {
        PyArrayObject *xa = reinterpret_cast<PyArrayObject*>($input);
        if ( PyArray_TYPE(xa) == NPY_INT ) {
            if ( PyArray_NDIM(xa) == 2 ) {
                $1 = PyArray_DIM(xa,0);
                $2 = PyArray_DIM(xa,1);
                $3  = static_cast<int*>(PyArray_DATA(xa));
            } else {
                SWIG_exception(SWIG_ValueError, "2d Array expected");
            }
        } else {
            SWIG_exception(SWIG_TypeError, "Array of integers expected");
        }
    } else {
        SWIG_exception(SWIG_TypeError, "Array expected");
    }
}

//-----------------------------------------------------------------------------
// Cleaner of temporary data when passing 2D NumPy arrays to C++ functions
// expecting double **
//-----------------------------------------------------------------------------
%{
namespace __private {
  class DppDeleter {
  public:
    double** amat;
    DppDeleter () {amat = 0;}
    ~DppDeleter ()
    {
      delete[] amat;
      //free(amat);
      amat = 0;
    }
  };
}
%}

//-----------------------------------------------------------------------------
// Typemap for 2D NumPy arrays to C++ functions expecting double **
//-----------------------------------------------------------------------------
%typemap(in) double**
{
    if PyArray_Check($input) {
        PyArrayObject *xa = reinterpret_cast<PyArrayObject*>($input);
        if ( PyArray_TYPE(xa) == NPY_DOUBLE ) {
            if ( PyArray_NDIM(xa) == 2 ) {
	        const int m = PyArray_DIM(xa,0);
	        const int n = PyArray_DIM(xa,1);
                $1 = new double*[m];
                double *data = reinterpret_cast<double*>((*xa).data);
                for (int i=0;i<m;++i)
                    $1[i] = &data[i*n];
            } else {
                SWIG_exception(SWIG_ValueError, "2d Array expected");
            }
        } else {
            SWIG_exception(SWIG_TypeError, "Array of doubles expected");
        }
    } else {
        SWIG_exception(SWIG_TypeError, "Array expected");
    }
}

//-----------------------------------------------------------------------------
// Delete temporary data
//-----------------------------------------------------------------------------
%typemap(freearg) double**
{
  delete[] $1;
}

//-----------------------------------------------------------------------------
// Typemap for 2D NumPy arrays to C++ functions expecting double **
//-----------------------------------------------------------------------------
%typemap(in) (int _matrix_dim_0, int _matrix_dim_1, double** _matrix) (__private::DppDeleter tmp)
{
  if PyArray_Check($input)
  {
    PyArrayObject *xa = reinterpret_cast<PyArrayObject *>($input);
    if ( PyArray_TYPE(xa) == NPY_DOUBLE )
    {
      if ( PyArray_NDIM(xa) == 2 )
      {
        int n = PyArray_DIM(xa,0);
        int m = PyArray_DIM(xa,1);
        $1 = n;
        $2 = m;
        double **amat = static_cast<double **>(malloc(n*sizeof*amat));
        double *data = reinterpret_cast<double *>(PyArray_DATA(xa));
        for (int i=0;i<n;++i)
            amat[i] = data + i*n;
        $3 = amat;
        tmp.amat = amat;
      }
      else
      {
        SWIG_exception(SWIG_ValueError, "2d Array expected");
      }
    }
    else
    {
      SWIG_exception(SWIG_TypeError, "Array of doubles expected");
    }
  }
  else
  {
    SWIG_exception(SWIG_TypeError, "Array expected");
  }
}

// vim:ft=cpp:
