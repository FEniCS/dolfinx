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
// First added:  2009-08-31
// Last changed: 2014-12-15

//=============================================================================
// In this file we declare what types that should be able to be passed using a
// std::vector typemap.
//
// We want to avoid using SWIGs own typemaps in std_vector.i,
// as we really just want to be able to pass argument, in and a out, using
// std::vector. We do not want to work with a proxy type of std::vector<Foo>,
// as the interface reflects the C++ type and is hence not 'pythonic'.
//=============================================================================

//-----------------------------------------------------------------------------
// Declare a dummy vector class
// This makes SWIG aware of the template type
//-----------------------------------------------------------------------------
namespace std
{
  template <typename T> class array
  {
  };
}

//-----------------------------------------------------------------------------
// Macro for defining an argout typemap for a std::vector of primitives
// The typemap returns a NumPy array of the primitive
//
// TYPE       : The primitive type
// TYPE_UPPER : The SWIG specific name of the type used in the array type checks
//              values SWIG use: INT32 for integer, DOUBLE for double aso.
// ARG_NAME   : The name of the argument that will be maped as an 'argout' argument
// NUMPY_TYPE : The type of the NumPy array that will be returned
//-----------------------------------------------------------------------------
%define ARGOUT_TYPEMAP_STD_ARRAY_OF_PRIMITIVES(TYPE, DIM, TYPE_UPPER, \
                                               ARG_NAME, NUMPY_TYPE)

//-----------------------------------------------------------------------------
// In typemap removing the argument from the expected in list
//-----------------------------------------------------------------------------
%typemap (in,numinputs=0) std::array<TYPE, DIM>& ARG_NAME (std::array<TYPE, DIM> array_temp)
{
  // ARGOUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(TYPE, TYPE_UPPER, ARG_NAME, NUMPY_TYPE)
  $1 = &array_temp;
}

//-----------------------------------------------------------------------------
// Argout typemap, returning a NumPy array for the std::vector<TYPE>
//-----------------------------------------------------------------------------
%typemap(argout) std::array<TYPE, DIM>& ARG_NAME
{
  npy_intp size = $1->size();
  PyArrayObject *ret = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size, NUMPY_TYPE));
  TYPE* data = static_cast<TYPE*>(PyArray_DATA(ret));
  for (int i = 0; i < size; ++i)
    data[i] = (*$1)[i];

  // Append the output to $result
  %append_output(PyArray_Return(ret));
}

%enddef

//-----------------------------------------------------------------------------
// Macro for defining an in typemap for a std::vector of primitives passed by
// value
//
// TYPE       : The primitive type
// TYPE_UPPER : The SWIG specific name of the type used in the array type checks
//              values SWIG use: INT32 for integer, DOUBLE for double aso.
// ARG_NAME   : The name of the argument that will be maped as an 'argout'
//              argument
// TYPE_NAME  : The name of the pointer type, 'double' for 'double', 'uint' for
//              'dolfin::uint'
// SEQ_LENGTH : An optional sequence length argument. If set to a negative
//              number
//              will no length check be made
//-----------------------------------------------------------------------------
%define PY_SEQUENCE_OF_SCALARS_TO_ARRAY_OF_PRIMITIVES(TYPE, DIM, TYPE_UPPER, \
                                                      ARG_NAME, TYPE_NAME, SEQ_LENGTH)

%typecheck(SWIG_TYPECHECK_ ## TYPE_UPPER ## _ARRAY) std::array<TYPE, DIM> ARG_NAME
{
  $1 = PySequence_Check($input) ? 1 : 0;
}

%typemap (in, fragment=Py_convert_frag(TYPE_NAME)) std::array<TYPE, DIM> ARG_NAME
  (std::array<TYPE, DIM> tmp_array, PyObject* item, TYPE value, std::size_t i)
{
  // PY_SEQUENCE_OF_SCALARS_TO_VECTOR_OF_PRIMITIVES(TYPE, TYPE_UPPER,
  //                                    ARG_NAME, TYPE_NAME, SEQ_LENGTH)

  // A first sequence test
  if (!PySequence_Check($input))
  {
    SWIG_exception(SWIG_TypeError, "expected a sequence for argument $argnum");
  }

  // Get sequence length
  Py_ssize_t pyseq_length = PySequence_Size($input);
  if (SEQ_LENGTH >= 0 && pyseq_length > SEQ_LENGTH)
  {
    SWIG_exception(SWIG_TypeError, "expected a sequence with length "	\
		   "SEQ_LENGTH for argument $argnum");
  }

  //tmp_array.reserve(pyseq_length);
  for (i = 0; i < pyseq_length; i++)
  {
    item = PySequence_ITEM($input, i);
    if(!SWIG_IsOK(Py_convert_ ## TYPE_NAME(item, value)))
    {
      Py_DECREF(item);
      SWIG_exception(SWIG_TypeError, "expected items of sequence to be of type "\
		     "\"TYPE_NAME\" in argument $argnum");
    }
    tmp_array[i] = value;
    Py_DECREF(item);
  }
  $1 = tmp_vec;
}
%enddef

//-----------------------------------------------------------------------------
// Macro for out typemaps of primitives of const std::vector<TYPE>& It returns
// readonly NumPy array
//
// TYPE       : The primitive type
// NUMPY_TYPE : The corresponding NumPy type
//-----------------------------------------------------------------------------
%define OUT_TYPEMAP_STD_ARRAY_OF_PRIMITIVES(TYPE, DIM, NUMPY_TYPE)

%typemap(out) std::array<TYPE, DIM>
{
  // OUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(TYPE, NUMPY_TYPE)
  npy_intp adims = $1.size();

  $result = PyArray_SimpleNew(1, &adims, NUMPY_TYPE);
  TYPE* data = static_cast<TYPE*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>($result)));
  std::copy($1.begin(), $1.end(), data);

}

%typemap(out) const std::array<TYPE, DIM> &
{
  // OUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(TYPE, NUMPY_TYPE) const std::vector<TYPE> &
  npy_intp adims = $1->size();

  $result = PyArray_SimpleNew(1, &adims, NUMPY_TYPE);
  TYPE* data = static_cast<TYPE*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>($result)));
  std::copy($1->begin(), $1->end(), data);
}

%enddef

//-----------------------------------------------------------------------------
// Macro for out typemaps of primitives of std::vector<TYPE> It returns a
// NumPy array vith a view. This is writable for const vectors and writable for
// non-const ones.
//
// TYPE      : The primitive type
// TYPE_NAME : The name of the pointer type, 'double' for 'double', 'uint' for
//             'dolfin::uint'
//-----------------------------------------------------------------------------
%define OUT_TYPEMAP_STD_ARRAY_OF_PRIMITIVES_REFERENCE(TYPE, DIM, TYPE_NAME)

%typemap(out, fragment=make_numpy_array_frag(1, TYPE_NAME)) const std::array<TYPE, DIM>&
{
  // OUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES_REFERENCE(TYPE, TYPE_NAME) const version
  $result = %make_numpy_array(1, TYPE_NAME)($1->size(), &($1->operator[](0)), false);
}

%typemap(out, fragment=make_numpy_array_frag(1, TYPE_NAME)) std::array<TYPE, DIM>&
{
  // OUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES_REFERENCE(TYPE, TYPE_NAME)
  $result = %make_numpy_array(1, TYPE_NAME)($1->size(), &($1->operator[](0)), true);
}

%enddef

//-----------------------------------------------------------------------------
// Run the different macros and instantiate the typemaps
//-----------------------------------------------------------------------------
ARGOUT_TYPEMAP_STD_ARRAY_OF_PRIMITIVES(double, 3, DOUBLE, , NPY_DOUBLE)
OUT_TYPEMAP_STD_ARRAY_OF_PRIMITIVES(double, 3, NPY_DOUBLE)
