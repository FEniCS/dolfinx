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
// Last changed: 2011-08-10

//=============================================================================
// In this file we declare some typemaps for the std::set type
//=============================================================================

namespace std
{
  template <class T> class set 
  {
  };
}


//-----------------------------------------------------------------------------
// Macro for defining an argout typemap for a std::set of primitives
// The typemaps makes a function returning a NumPy array of that primitive
//
// TYPE       : The primitive type
// TYPE_UPPER : The SWIG specific name of the type used in the array type checks
//              values SWIG use: INT32 for integer, DOUBLE for double aso.
// ARG_NAME   : The name of the argument that will be maped as an 'argout' argument
// NUMPY_TYPE : The type of the NumPy array that will be returned
//-----------------------------------------------------------------------------
%define ARGOUT_TYPEMAP_BOOST_UNORDERED_SET_OF_PRIMITIVES(TYPE, TYPE_UPPER, ARG_NAME, NUMPY_TYPE)
//-----------------------------------------------------------------------------
// In typemap removing the argument from the expected in list
//-----------------------------------------------------------------------------
%typemap (in,numinputs=0) std::set<TYPE>& ARG_NAME (std::set<TYPE> set_temp)
{
  $1 = &set_temp;
}

//-----------------------------------------------------------------------------
// Argout typemap, returning a NumPy array for the boost::unordered_set<TYPE>
//-----------------------------------------------------------------------------
%typemap(argout) std::set<TYPE> & ARG_NAME
{
  PyObject* o0 = 0;
  PyObject* o1 = 0;
  PyObject* o2 = 0;
  npy_intp size = $1->size();
  PyArrayObject *ret = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size, NUMPY_TYPE));
  TYPE* data = static_cast<TYPE*>(PyArray_DATA(ret));

  int i = 0;
  for (std::set<TYPE>::const_iterator it = (*$1).begin(); it != (*$1).end(); ++it) 
  {
    data[i] = *it;
    ++i;
  }
  o0 = PyArray_Return(ret);
  // If the $result is not already set
  if ((!$result) || ($result == Py_None)) 
  {
    $result = o0;
  }
  // If the result is set by another out typemap build a tuple of arguments
  else
  {
    // If the the argument is set but is not a tuple make one and put the result in it
    if (!PyTuple_Check($result)) 
    {
      o1 = $result;
      $result = PyTuple_New(1);
      PyTuple_SetItem($result, 0, o1);
    }
    o2 = PyTuple_New(1);
    PyTuple_SetItem(o2, 0, o0);
    o1 = $result;
    $result = PySequence_Concat(o1, o2);
    Py_DECREF(o1);
    Py_DECREF(o2);
  }
}

%enddef

ARGOUT_TYPEMAP_BOOST_UNORDERED_SET_OF_PRIMITIVES(dolfin::uint, INT32, ids_result, NPY_INT)
ARGOUT_TYPEMAP_BOOST_UNORDERED_SET_OF_PRIMITIVES(dolfin::uint, INT32, cells, NPY_INT)
