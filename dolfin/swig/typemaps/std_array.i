/* -*- C -*- */
// Copyright (C) 2009-2016 Johan Hake and Garth N. Wells
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

//=============================================================================
// In this file we declare what types that should be able to be passed
// using a std::array typemap. It is modelled on std_vector.i
// =============================================================================

//-----------------------------------------------------------------------------
// Declare a dummy array class. This makes SWIG aware of the template
// type
// -----------------------------------------------------------------------------
namespace std
{
  template <typename T> class array
  {
  };
}

//-----------------------------------------------------------------------------
// Macro for defining an in typemap for a std::array of primitives passed by
// value
//
// TYPE       : The primitive type
// TYPE_UPPER : The SWIG specific name of the type used in the array type checks
//              values SWIG use: INT32 for integer, DOUBLE for double aso.
// ARG_NAME   : The name of the argument that will be maped as an 'argout'
//              argument
// TYPE_NAME  : The name of the pointer type, 'double' for 'double', 'uint' for
//              'dolfin::uint'
//-----------------------------------------------------------------------------
%define PY_SEQUENCE_OF_SCALARS_TO_ARRAY_OF_PRIMITIVES(TYPE, DIM, TYPE_UPPER, \
                                                      ARG_NAME, TYPE_NAME)

%typecheck(SWIG_TYPECHECK_ ## TYPE_UPPER ## _ARRAY) std::array<TYPE, DIM> ARG_NAME
{ $1 = PySequence_Check($input) ? 1 : 0; }

%typemap (in, fragment=Py_convert_frag(TYPE_NAME)) std::array<TYPE, DIM> ARG_NAME
  (std::array<TYPE, DIM> tmp_array, PyObject* item, TYPE value, std::size_t i)
{
  // A first sequence test
  if (!PySequence_Check($input))
  {
    SWIG_exception(SWIG_TypeError, "expected a sequence for argument $argnum");
  }

  // Get sequence length
  Py_ssize_t pyseq_length = PySequence_Size($input);
  if (pyseq_length != DIM)
  {
    SWIG_exception(SWIG_TypeError, "expected a sequence with length DIM");
  }

  for (i = 0; i < pyseq_length; i++)
  {
    item = PySequence_ITEM($input, i);
    if(!SWIG_IsOK(Py_convert_ ## TYPE_NAME(item, value)))
    {
      Py_DECREF(item);
      SWIG_exception(SWIG_TypeError, "expected items of sequence to be of type " \
		     "\"TYPE_NAME\" in argument $argnum");
    }
    tmp_array[i] = value;
    Py_DECREF(item);
  }
  $1 = tmp_array;
}
%enddef

//-----------------------------------------------------------------------------
// Macro for out typemaps of primitives of std::array<TYPE, DIM> It returns
// a NumPy array
//
// TYPE       : The primitive type
// NUMPY_TYPE : The corresponding NumPy type
//-----------------------------------------------------------------------------
%define OUT_TYPEMAP_STD_ARRAY_OF_PRIMITIVES(TYPE, DIM, NUMPY_TYPE)

%typemap(out) std::array<TYPE, DIM>
{
  npy_intp adims = $1.size();

  $result = PyArray_SimpleNew(1, &adims, NUMPY_TYPE);
  TYPE* data = static_cast<TYPE*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>($result)));
  std::copy($1.begin(), $1.end(), data);
}

%enddef

//-----------------------------------------------------------------------------
// Run the different macros and instantiate the typemaps
//-----------------------------------------------------------------------------
OUT_TYPEMAP_STD_ARRAY_OF_PRIMITIVES(double, 3, NPY_DOUBLE)
PY_SEQUENCE_OF_SCALARS_TO_ARRAY_OF_PRIMITIVES(double, 3, DOUBLE, rgb, double)
