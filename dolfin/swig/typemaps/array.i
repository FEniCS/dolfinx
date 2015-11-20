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
// First added:  2009-12-10
// Last changed: 2011-06-15

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
// TYPE_NAME  : The name of the pointer type, 'double' for 'double', 'uint' for
//              'unsigned int'
// ARGNAME    : The name of the argument the typemap will kick in for pass nothing
//              and the typemap will kick in for all argument names
// CONSTARRAY : If the dolfin::Array is of type const, then pass const for this
//              argument
//-----------------------------------------------------------------------------
%define IN_NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(TYPE, TYPECHECK, NUMPYTYPE, TYPE_NAME, ARGNAME, CONSTARRAY)

%typemap(in, fragment=convert_numpy_to_array_with_check(TYPE_NAME)) (CONSTARRAY dolfin::Array<TYPE> &ARGNAME) (std::size_t size, TYPE* data)
{
  if (!convert_numpy_to_array_with_check_ ## TYPE_NAME($input, size, data))
    return NULL;
  $1 = new dolfin::Array<TYPE>(size, data);
}

// Clean up the constructed Array
%typemap(freearg) CONSTARRAY dolfin::Array<TYPE>& ARGNAME
{
  delete $1;
}

%typecheck(SWIG_TYPECHECK_ ## TYPECHECK ## _ARRAY) CONSTARRAY dolfin::Array<TYPE> & ARGNAME
{
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

%typemap(out) dolfin::Array<TYPE>
{
  // Create NumPy array
  npy_intp size = (&$1)->size();
  PyObject* op = PyArray_SimpleNew(1, &size, NUMPYTYPE);

  if (op == NULL)
    SWIG_exception(SWIG_TypeError, "Error in conversion of dolfin::Array< TYPE > to NumPy array.");

  // Get data
  TYPE* data = reinterpret_cast<TYPE*>(PyArray_DATA(op));

  // Set data from Array
  for (int i = 0; i < (&$1)->size(); i++)
    data[i] = (&$1)->operator[](i);

  // Return the NumPy array
  $result = op;
}
%enddef

//-----------------------------------------------------------------------------
// Director typemaps for dolfin::Array
//-----------------------------------------------------------------------------
%typemap(directorin) const dolfin::Array<double>&
{
  $input = %make_numpy_array(1, double)($1_name.size(), $1_name.data(), false);
}

%typemap(directorin) dolfin::Array<double>&
{
  $input = %make_numpy_array(1, double)($1_name.size(), $1_name.data(), true);
}

//-----------------------------------------------------------------------------
// Run the typemap macros
//-----------------------------------------------------------------------------

// Instantiate argument name specific typemaps for non const arguments
IN_NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(double, DOUBLE, NPY_DOUBLE, double, values, )
IN_NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(double, DOUBLE, NPY_DOUBLE, double, y, )
IN_NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(std::size_t, INT32, NPY_UINTP, size_t, indices,)
IN_NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(double, DOUBLE, NPY_DOUBLE, double, vertex_values, )
IN_NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(dolfin::la_index, INT32, NPY_UINTP, dolfin_index, indices,)

// Instantiate argument name independent typemaps for all
// const Array <{int, uint, double}>& arguments
IN_NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(double, DOUBLE, NPY_DOUBLE, double, , const)
IN_NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(std::size_t, INT32, NPY_UINTP, size_t, , const)
IN_NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(int, INT32, NPY_INT, int, , const)
IN_NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(dolfin::la_index, INT32, NPY_UINT, dolfin_index, , const)

OUT_NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(std::size_t, NPY_UINTP)
OUT_NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(int, NPY_INT)
OUT_NUMPY_TYPEMAP_FOR_DOLFIN_ARRAY(double, NPY_DOUBLE)


/*
// Out typemap for ArrayView<const dolfin::la_index>. ArrayView is
// used mainly inside the library.
%typemap(out) dolfin::ArrayView<const dolfin::la_index>
{
  npy_intp adims = $1.size();

  #if (DOLFIN_LA_INDEX_SIZE==4)
  $result = PyArray_SimpleNew(1, &adims, NPY_INT);
  int* data = static_cast<int*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>($result)));
  #else
  $result = PyArray_SimpleNew(1, &adims, NPY_INT64);
  int64_t* data = static_cast<int64_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>($result)));
  #endif

  // Copy data
  std::copy($1.begin(), $1.end(), data);
}
*/
