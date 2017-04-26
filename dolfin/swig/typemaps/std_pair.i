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
// Last changed: 2014-04-07

//-----------------------------------------------------------------------------
// User macro for defining in typemaps for std::pair of a pointer to some
// DOLFIN type and a double
//-----------------------------------------------------------------------------
%define IN_TYPEMAPS_STD_PAIR_OF_POINTER_AND_DOUBLE(TYPE)

//-----------------------------------------------------------------------------
// Make SWIG aware of the shared_ptr version of TYPE
//-----------------------------------------------------------------------------
%types(std::shared_ptr<TYPE >*);
//
////-----------------------------------------------------------------------------
//// Run the macros for the combination of const and no const of
//// {const} std::vector<{const} dolfin::TYPE *>
////-----------------------------------------------------------------------------
////IN_TYPEMAP_STD_VECTOR_OF_POINTERS(TYPE,,)
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

  // TYPECHEK IN_TYPEMAP_STD_PAIR_OF_POINTER_AND_DOUBLE(TYPE, CONST, CONST_PAIR)
  $1 = 0;
  if (PyTuple_Check($input) && PyTuple_Size($input) == 2)
  {
    void* itemp = 0;
    PyObject* py_first  = PyTuple_GetItem($input, 0);
    int res = SWIG_ConvertPtr(py_first, &itemp, $descriptor(dolfin::TYPE *), 0);
    if (!SWIG_IsOK(res))
    {
      // If failed with normal pointer conversion then
      // try with shared_ptr conversion
      int newmem = 0;
      res = SWIG_ConvertPtrAndOwn(py_first, &itemp, $descriptor(std::shared_ptr< dolfin::TYPE > *), 0, &newmem);
      if (SWIG_IsOK(res))
      {
	$1 = 1;
	if (newmem & SWIG_CAST_NEW_MEMORY)
	{
	  delete reinterpret_cast< std::shared_ptr< dolfin::TYPE > * >(itemp);
	}
      }
    }
    else
    {
      $1 = 1;
    }
  }
}

//-----------------------------------------------------------------------------
// The typemap
//-----------------------------------------------------------------------------
%typemap(in) CONST_PAIR std::pair<CONST dolfin::TYPE*, double> (std::pair<CONST dolfin::TYPE*, double> tmp_pair, std::shared_ptr<dolfin::TYPE > tempshared, dolfin::TYPE * arg)
{
  // TYPEMAP IN_TYPEMAP_STD_PAIR_OF_POINTER_AND_DOUBLE(TYPE, CONST, CONST_PAIR)
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
    res = SWIG_ConvertPtrAndOwn(py_first, &itemp, $descriptor(std::shared_ptr< dolfin::TYPE > *), 0, &newmem);
    if (SWIG_IsOK(res)){
      // If we need to release memory
      if (newmem & SWIG_CAST_NEW_MEMORY)
      {
        tempshared = *reinterpret_cast< std::shared_ptr<dolfin::TYPE > * >(itemp);
        delete reinterpret_cast< std::shared_ptr< dolfin::TYPE > * >(itemp);
        arg = const_cast< dolfin::TYPE * >(tempshared.get());
      }
      else
      {
        arg = const_cast< dolfin::TYPE * >(reinterpret_cast< std::shared_ptr< dolfin::TYPE > * >(itemp)->get());
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
IN_TYPEMAPS_STD_PAIR_OF_POINTER_AND_DOUBLE(Mesh)
IN_TYPEMAPS_STD_PAIR_OF_POINTER_AND_DOUBLE(MeshFunction<int>)
IN_TYPEMAPS_STD_PAIR_OF_POINTER_AND_DOUBLE(MeshFunction<std::size_t> )
IN_TYPEMAPS_STD_PAIR_OF_POINTER_AND_DOUBLE(MeshFunction<double> )
IN_TYPEMAPS_STD_PAIR_OF_POINTER_AND_DOUBLE(MeshFunction<bool>)
IN_TYPEMAPS_STD_PAIR_OF_POINTER_AND_DOUBLE(Point)

//-----------------------------------------------------------------------------
// In typemap for std::pair<TYPE,TYPE>
//-----------------------------------------------------------------------------
%typecheck(SWIG_TYPECHECK_POINTER) std::pair<std::size_t, std::size_t>
{
  $1 = PyTuple_Check($input) ? 1 : 0;
}

%typemap(in) std::pair<std::size_t, std::size_t> (std::pair<std::size_t, std::size_t> tmp_pair, long tmp)
{
  // Check that we have a tuple
  if (!PyTuple_Check($input) || PyTuple_Size($input) != 2)
    SWIG_exception(SWIG_TypeError, "expected a tuple of length 2 of positive integers.");

  // Get pointers to function and time
  PyObject* py_first  = PyTuple_GetItem($input, 0);
  PyObject* py_second = PyTuple_GetItem($input, 1);

  // Check and get first tuple value
  if (!Py_convert_std_size_t(py_first, tmp_pair.first))
    SWIG_exception(SWIG_TypeError, "expected positive 'int' as the first tuple argument ");

  // Check and get second tuple value
  if (!Py_convert_std_size_t(py_second, tmp_pair.second))
    SWIG_exception(SWIG_TypeError, "expected positive 'int' as the second tuple argument ");

  // Assign input variable
  $1 = tmp_pair;

}


%typecheck(SWIG_TYPECHECK_POINTER) std::pair<std::int64_t, std::int64_t>
{
  $1 = PyTuple_Check($input) ? 1 : 0;
}
%typemap(in) std::pair<std::int64_t, std::int64_t> (std::pair<std::int64_t, std::int64_t> tmp_pair, long tmp)
{
  // Check that we have a tuple
  if (!PyTuple_Check($input) || PyTuple_Size($input) != 2)
    SWIG_exception(SWIG_TypeError, "expected a tuple of length 2 of ints.");

  // Get pointers to function and time
  PyObject* py_first  = PyTuple_GetItem($input, 0);
  PyObject* py_second = PyTuple_GetItem($input, 1);

  tmp_pair = std::make_pair(PyLong_AsLongLong(py_first), PyLong_AsLongLong(py_second));

  // Assign input variable
  $1 = tmp_pair;
}


%typecheck(SWIG_TYPECHECK_POINTER) std::pair<double, double>
{
  $1 = PyTuple_Check($input) ? 1 : 0;
}

%typemap(in) std::pair<double, double> (std::pair<double, double> tmp_pair, long tmp)
{
  // Check that we have a tuple
  if (!PyTuple_Check($input) || PyTuple_Size($input) != 2)
    SWIG_exception(SWIG_TypeError, "expected a tuple of length 2 of floats.");

  // Get pointers to function and time
  PyObject* py_first  = PyTuple_GetItem($input, 0);
  PyObject* py_second = PyTuple_GetItem($input, 1);

  tmp_pair = std::make_pair(PyFloat_AsDouble(py_first), PyFloat_AsDouble(py_second));

  // Assign input variable
  $1 = tmp_pair;
}

//-----------------------------------------------------------------------------
// Out typemap for std::pair<TYPE,TYPE>
//-----------------------------------------------------------------------------
%typemap(out) std::pair<std::size_t, std::size_t>
{ $result = Py_BuildValue("ii", $1.first, $1.second); }

%typemap(out) std::pair<std::int64_t, std::int64_t>
{ $result = Py_BuildValue("LL", $1.first, $1.second); }

%typemap(out) std::pair<std::size_t, bool>
{ $result = Py_BuildValue("ib", $1.first, $1.second); }

%typemap(out) std::pair<unsigned int, unsigned int>
{ $result = Py_BuildValue("ii", $1.first, $1.second); }

%typemap(out) std::pair<unsigned int, bool>
{ $result = Py_BuildValue("ib", $1.first, $1.second); }

%typemap(out) std::pair<double, double>
{ $result = Py_BuildValue("dd", $1.first, $1.second); }

%typemap(out) std::pair<unsigned int, double>
{ $result = Py_BuildValue("id", $1.first, $1.second); }


// FIXME: Add macro for the two typemaps below

//-----------------------------------------------------------------------------
// Out typemap for std::pair<std::vector<unsigned int>, std::vector<unsigned int> >
// If we need should need it for other types, we can make it into a macro later.
//-----------------------------------------------------------------------------
%typemap(out) std::pair<std::vector<unsigned int>, std::vector<unsigned int> >
{
  npy_intp n0 = $1.first.size();
  npy_intp n1 = $1.second.size();

  PyArrayObject *x0 = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &n0, NPY_UINT));
  PyArrayObject *x1 = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &n1, NPY_UINT));

  unsigned int* data0 = static_cast<unsigned int*>(PyArray_DATA(x0));
  unsigned int* data1 = static_cast<unsigned int*>(PyArray_DATA(x1));

  std::copy($1.first.begin(),  $1.first.end(),  data0);
  std::copy($1.second.begin(), $1.second.end(), data1);

  $result = Py_BuildValue("OO", x0, x1);
}

//-----------------------------------------------------------------------------
// Out typemap for std::pair<std::vector<double>, std::vector<double> >
// If we need should need it for other types, we can make it into a macro later.
//-----------------------------------------------------------------------------
%typemap(out) std::pair<std::vector<double>, std::vector<double> >
{
  npy_intp n0 = $1.first.size();
  npy_intp n1 = $1.second.size();

  PyArrayObject *x0 = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &n0, NPY_DOUBLE));
  PyArrayObject *x1 = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &n1, NPY_DOUBLE));

  double* data0 = static_cast<double*>(PyArray_DATA(x0));
  double* data1 = static_cast<double*>(PyArray_DATA(x1));

  std::copy($1.first.begin(),  $1.first.end(),  data0);
  std::copy($1.second.begin(), $1.second.end(), data1);

  $result = Py_BuildValue("OO", x0, x1);
}
