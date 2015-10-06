/* -*- C -*- */
// Copyright (C) 2009 Andre Massing
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
// First added:  2009-11-27
// Last changed: 2013-06-24

//=============================================================================
// In this file we declare some typemaps for the std::set type
//=============================================================================

namespace std
{
  template <typename T> class set
  {
  };
}


//-----------------------------------------------------------------------------
// Macro for defining an argout typemap for a std::set of primitives
// The typemaps makes a function returning a NumPy array of that primitive
//
// TYPE       : The primitive type
// ARG_NAME   : The name of the argument that will be maped as an 'argout' argument
// NUMPY_TYPE : The type of the NumPy array that will be returned
//-----------------------------------------------------------------------------
%define ARGOUT_TYPEMAP_STD_SET_OF_PRIMITIVES(TYPE, ARG_NAME, NUMPY_TYPE)
//-----------------------------------------------------------------------------
// In typemap removing the argument from the expected in list
//-----------------------------------------------------------------------------
%typemap (in,numinputs=0) std::set<TYPE>& ARG_NAME (std::set<TYPE> set_temp)
{
  $1 = &set_temp;
}

//-----------------------------------------------------------------------------
// Argout typemap, returning a NumPy array for the std::unordered_set<TYPE>
//-----------------------------------------------------------------------------
%typemap(argout) std::set<TYPE> & ARG_NAME
{
  npy_intp size = $1->size();
  PyArrayObject *ret
    = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size, NUMPY_TYPE));
  TYPE* data = static_cast<TYPE*>(PyArray_DATA(ret));

  int i = 0;
  for (std::set<TYPE>::const_iterator it = (*$1).begin(); it != (*$1).end();
       ++it)
  {
    data[i] = *it;
    ++i;
  }

  // Append the output to $result
  %append_output(PyArray_Return(ret));
}

%enddef

//-----------------------------------------------------------------------------
// Macro for defining an out typemap for a std::unordered_set of primitives
// The typemaps makes a function returning a NumPy array of that primitive
//
// TYPE       : The primitive type
// NUMPY_TYPE : The type of the NumPy array that will be returned
//-----------------------------------------------------------------------------
%define OUT_SET_TYPEMAP_OF_PRIMITIVES(TYPE, NUMPY_TYPE)
SET_SPECIFIC_OUT_TYPEMAP_OF_PRIMITIVES(std::unordered_map, TYPE, NUMPY_TYPE)
SET_SPECIFIC_OUT_TYPEMAP_OF_PRIMITIVES(std::set, TYPE, NUMPY_TYPE)
%enddef


//-----------------------------------------------------------------------------
// Argout typemap, returning a NumPy array for the std::unordered_set<TYPE>
//-----------------------------------------------------------------------------
%define SET_SPECIFIC_OUT_TYPEMAP_OF_PRIMITIVES(SET_TYPE, TYPE, NUMPY_TYPE)

// Value version
%typemap(out) SET_TYPE<TYPE>
{
  npy_intp size = $1.size();
  PyArrayObject *ret
    = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size, NUMPY_TYPE));
  TYPE* data = static_cast<TYPE*>(PyArray_DATA(ret));

  int i = 0;
  for (SET_TYPE<TYPE>::const_iterator it = $1.begin(); it != $1.end(); ++it)
  {
    data[i] = *it;
    ++i;
  }

  // Append the output to $result
  $result = PyArray_Return(ret);
}

// Reference version
%typemap(out) const SET_TYPE<TYPE> &
{
  npy_intp size = $1->size();
  PyArrayObject *ret
    = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size, NUMPY_TYPE));
  TYPE* data = static_cast<TYPE*>(PyArray_DATA(ret));

  int i = 0;
  for (SET_TYPE<TYPE>::const_iterator it = $1->begin(); it != $1->end();
       it++, i++)
  {
    data[i] = *it;
  }

  // Append the output to $result
  $result = PyArray_Return(ret);
}

%enddef

// NOTE: SWIG BUG
// NOTE: Because of bug introduced by SWIG 2.0.5 we cannot use
// templated versions
// NOTE: of typdefs, which means we need to use unsigned int instead
// of dolfin::uint NOTE: in typemaps
ARGOUT_TYPEMAP_STD_SET_OF_PRIMITIVES(std::size_t, ids_result, NPY_UINTP)
ARGOUT_TYPEMAP_STD_SET_OF_PRIMITIVES(std::size_t, cells, NPY_UINTP)
OUT_SET_TYPEMAP_OF_PRIMITIVES(std::size_t, NPY_UINTP)
OUT_SET_TYPEMAP_OF_PRIMITIVES(int, NPY_INT)
OUT_SET_TYPEMAP_OF_PRIMITIVES(unsigned int, NPY_UINT)


// ---------------------------------------------------------------------------
// Typemaps (in) for std::set<dolfin::TimingType>
// ---------------------------------------------------------------------------
%typecheck(SWIG_TYPECHECK_INT32_ARRAY) std::set<dolfin::TimingType>
{
    $1 = PySequence_Check($input) ? 1 : 0;
}

%typemap(in) std::set<dolfin::TimingType> (std::set<dolfin::TimingType> tmp)
{
  if (!PyList_Check($input)) {
    PyErr_SetString(PyExc_ValueError,"expected a list of 'TimingType' (aka int32).");
    return NULL;
  }
  int list_length = PyList_Size($input);
  if (!list_length > 0){
    PyErr_SetString(PyExc_ValueError,"expected a list with length > 0");
    return NULL;
  }
  for (int i = 0; i < list_length; i++)
  {
    PyObject *o = PyList_GetItem($input,i);
%#if PY_VERSION_HEX>=0x03000000
    if (PyLong_Check(o))
%#else
    if (PyInt_Check(o))
%#endif
    {
      int int_o = PyInt_AsLong(o);
      tmp.insert(static_cast<dolfin::TimingType>(int_o));
    }
    else
    {
      PyErr_SetString(PyExc_TypeError,"provide a list of 'TimingType' (aka int32).");
      return NULL;
    }
  }
  $1 = tmp;
}
