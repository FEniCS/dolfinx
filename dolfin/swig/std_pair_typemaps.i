/* -*- C -*- */
// Copyright (C) 2011 Johan Hake
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
// First added:  2011-01-25
// Last changed: 2011-01-25

//-----------------------------------------------------------------------------
// User macro for defining in typmaps for std::pair of a pointer to some
// DOLFIN type and a double
//-----------------------------------------------------------------------------
%define IN_TYPEMAPS_STD_PAIR_OF_POINTER_AND_DOUBLE(TYPE)

//-----------------------------------------------------------------------------
// Make SWIG aware of the shared_ptr version of TYPE
//-----------------------------------------------------------------------------
%types(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr<TYPE>*);

//-----------------------------------------------------------------------------
// Run the macros for the combination of const and no const of
// {const} std::vector<{const} dolfin::TYPE *>
//-----------------------------------------------------------------------------
//IN_TYPEMAP_STD_VECTOR_OF_POINTERS(TYPE,,)
IN_TYPEMAP_STD_PAIR_OF_POINTER_AND_DOUBLE(TYPE,const,)
IN_TYPEMAP_STD_PAIR_OF_POINTER_AND_DOUBLE(TYPE,,const)
IN_TYPEMAP_STD_PAIR_OF_POINTER_AND_DOUBLE(TYPE,const,const)

%enddef

//-----------------------------------------------------------------------------
// Macro for defining in typemaps for
// {const} std::vector<{const} dolfin::TYPE *>
// using a Python List of TYPE
//-----------------------------------------------------------------------------
%define IN_TYPEMAP_STD_PAIR_OF_POINTER_AND_DOUBLE(TYPE, CONST, CONST_PAIR)

//-----------------------------------------------------------------------------
// The typecheck
//-----------------------------------------------------------------------------
%typecheck(SWIG_TYPECHECK_POINTER) CONST_PAIR std::pair<CONST dolfin::TYPE *, double>
{
  $1 = PyTuple_Check($input) ? 1 : 0;
}

//-----------------------------------------------------------------------------
// The typemap
//-----------------------------------------------------------------------------
%typemap(in) CONST_PAIR std::pair<CONST dolfin::TYPE*, double> (std::pair<CONST dolfin::TYPE*, double> tmp_pair, SWIG_SHARED_PTR_QNAMESPACE::shared_ptr<dolfin::TYPE> tempshared, dolfin::TYPE * arg)
{
  int res = 0;
  void* itemp = 0;
  int newmem = 0;

  // Check that we have a tuple
  if (!PyTuple_Check($input) || PyTuple_Size($input) != 2)
    SWIG_exception(SWIG_TypeError, "expected a tuple of length 2 with TYPE and Float.");

  // Get pointers to function and time
  PyObject* py_first  = PyTuple_GetItem($input, 0);
  PyObject* py_second = PyTuple_GetItem($input, 1);

  // Check that we have a float
  if (!PyFloat_Check(py_second))
    SWIG_exception(SWIG_TypeError, "expected a Float for the second tuple argument.");

  // Get second variable
  tmp_pair.second = PyFloat_AsDouble(py_second);

  res = SWIG_ConvertPtr(py_first, &itemp, $descriptor(dolfin::TYPE *), 0);
  if (SWIG_IsOK(res))
  {
    tmp_pair.first = reinterpret_cast<dolfin::TYPE *>(itemp);
  }
  else{
    // If failed with normal pointer conversion then
    // try with shared_ptr conversion
    newmem = 0;
    res = SWIG_ConvertPtrAndOwn(py_first, &itemp, $descriptor(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::TYPE > *), 0, &newmem);
    if (SWIG_IsOK(res)){
      // If we need to release memory
      if (newmem & SWIG_CAST_NEW_MEMORY)
      {
        tempshared = *reinterpret_cast< SWIG_SHARED_PTR_QNAMESPACE::shared_ptr<dolfin::TYPE> * >(itemp);
        delete reinterpret_cast< SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::TYPE > * >(itemp);
        arg = const_cast< dolfin::TYPE * >(tempshared.get());
      }
      else
      {
        arg = const_cast< dolfin::TYPE * >(reinterpret_cast< SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::TYPE > * >(itemp)->get());
      }
      tmp_pair.first = arg;
    }
    else
    {
      SWIG_exception(SWIG_TypeError, "expected tuple of TYPE and Float (Bad conversion)");
    }
  }

  // Assign the input variable
  $1 = tmp_pair;
}
%enddef

//-----------------------------------------------------------------------------
// Run the different macros and instantiate the typemaps
//-----------------------------------------------------------------------------
IN_TYPEMAPS_STD_PAIR_OF_POINTER_AND_DOUBLE(Function)

//-----------------------------------------------------------------------------
// In typemap for std::pair<TYPE,TYPE>
//-----------------------------------------------------------------------------
%typecheck(SWIG_TYPECHECK_POINTER) std::pair<dolfin::uint, dolfin::uint>
{
  $1 = PyTuple_Check($input) ? 1 : 0;
}

%typemap(in) std::pair<dolfin::uint, dolfin::uint> (std::pair<dolfin::uint, dolfin::uint> tmp_pair, long tmp)
{
  // Check that we have a tuple
  if (!PyTuple_Check($input) || PyTuple_Size($input) != 2)
    SWIG_exception(SWIG_TypeError, "expected a tuple of length 2 of positive integers.");

  // Get pointers to function and time
  PyObject* py_first  = PyTuple_GetItem($input, 0);
  PyObject* py_second = PyTuple_GetItem($input, 1);

  // Check and get first tuple value
  if (PyInteger_Check(py_first))
  {
    tmp = static_cast<long>(PyInt_AsLong(py_first));
    if (tmp>=0)
       tmp_pair.first = static_cast<dolfin::uint>(tmp);
    else
      SWIG_exception(SWIG_TypeError, "expected positive 'int' as the first tuple argument ");
  }
  else
    SWIG_exception(SWIG_TypeError, "expected positive 'int' as the first tuple argument ");

  // Check and get second tuple value
  if (PyInteger_Check(py_second))
  {
    tmp = static_cast<long>(PyInt_AsLong(py_second));
    if (tmp>=0)
       tmp_pair.second = static_cast<dolfin::uint>(tmp);
    else
      SWIG_exception(SWIG_TypeError, "expected positive 'int' as the second tuple argument ");
  }
  else
    SWIG_exception(SWIG_TypeError, "expected positive 'int' as the second tuple argument ");

  // Assign input variable
  $1 = tmp_pair;

}

//-----------------------------------------------------------------------------
// Out typemap for std::pair<TYPE,TYPE>
//-----------------------------------------------------------------------------
%typemap(out) std::pair<dolfin::uint, dolfin::uint>
{
  $result = Py_BuildValue("ii",$1.first,$1.second);
}
%typemap(out) std::pair<dolfin::uint,bool>
{
  $result = Py_BuildValue("ib",$1.first,$1.second);
}
