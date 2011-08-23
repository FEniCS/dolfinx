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
// Last changed: 2011-08-19

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
  template <class T> class vector
  {
  };
}

//-----------------------------------------------------------------------------
// User macro for defining in typemaps for std::vector of pointers or
// shared_pointer to some DOLFIN type
//-----------------------------------------------------------------------------
%define TYPEMAPS_STD_VECTOR_OF_POINTERS(TYPE)

//-----------------------------------------------------------------------------
// Make SWIG aware of the shared_ptr version of TYPE
//-----------------------------------------------------------------------------
%types(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr<dolfin::TYPE>*);

//-----------------------------------------------------------------------------
// Run the macros for the combination of const and no const of
// {const} std::vector<{const} dolfin::TYPE *>
//-----------------------------------------------------------------------------
//IN_TYPEMAP_STD_VECTOR_OF_POINTERS(TYPE,,)
IN_TYPEMAP_STD_VECTOR_OF_POINTERS(TYPE,const,)
IN_TYPEMAP_STD_VECTOR_OF_POINTERS(TYPE,,const)
IN_TYPEMAP_STD_VECTOR_OF_POINTERS(TYPE,const,const)

%enddef

//-----------------------------------------------------------------------------
// Macro for defining in typemaps for
// {const} std::vector<{const} dolfin::TYPE *>
// using a Python List of TYPE
//-----------------------------------------------------------------------------
%define IN_TYPEMAP_STD_VECTOR_OF_POINTERS(TYPE,CONST,CONST_VECTOR)

//-----------------------------------------------------------------------------
// The typecheck
//-----------------------------------------------------------------------------
%typecheck(SWIG_TYPECHECK_POINTER) CONST_VECTOR std::vector<CONST dolfin::TYPE *> &
{
  $1 = PyList_Check($input) ? 1 : 0;
}

//-----------------------------------------------------------------------------
// The std::vector<Type*> typemap
//-----------------------------------------------------------------------------
%typemap (in) CONST_VECTOR std::vector<CONST dolfin::TYPE *> & (
std::vector<CONST dolfin::TYPE *> tmp_vec,
SWIG_SHARED_PTR_QNAMESPACE::shared_ptr<dolfin::TYPE> tempshared,
dolfin::TYPE * arg)
{
  // IN_TYPEMAP_STD_VECTOR_OF_POINTERS(TYPE, CONST, CONST_VECTOR)
  if (PyList_Check($input))
  {
    int size = PyList_Size($input);
    int res = 0;
    PyObject * py_item = 0;
    void * itemp = 0;
    int newmem = 0;
    tmp_vec.reserve(size);
    for (int i = 0; i < size; i++)
    {
      py_item = PyList_GetItem($input,i);
      res = SWIG_ConvertPtr(py_item, &itemp, $descriptor(dolfin::TYPE *), 0);
      if (SWIG_IsOK(res))
      {
        tmp_vec.push_back(reinterpret_cast<dolfin::TYPE *>(itemp));
      }
      else
      {
        // If failed with normal pointer conversion then
        // try with shared_ptr conversion
        newmem = 0;
        res = SWIG_ConvertPtrAndOwn(py_item, &itemp, $descriptor(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::TYPE > *), 0, &newmem);
        if (SWIG_IsOK(res))
        {
	  if (itemp)
	  {
	    tempshared = *(reinterpret_cast< SWIG_SHARED_PTR_QNAMESPACE::shared_ptr<dolfin::TYPE> * >(itemp));
	    tmp_vec.push_back(tempshared.get());
	  }
          // If we need to release memory
          if (newmem & SWIG_CAST_NEW_MEMORY)
          {
            delete reinterpret_cast< SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::TYPE > * >(itemp);
          }
        }
        else
        {
          SWIG_exception(SWIG_TypeError, "list of TYPE expected (Bad conversion)");
        }
      }
    }
    $1 = &tmp_vec;
  }
  else
  {
    SWIG_exception(SWIG_TypeError, "list of TYPE expected");
  }
}


//-----------------------------------------------------------------------------
// The std::vector<Type*> typecheck
//-----------------------------------------------------------------------------
%typecheck(SWIG_TYPECHECK_POINTER) CONST_VECTOR std::vector<SWIG_SHARED_PTR_QNAMESPACE::shared_ptr<CONST dolfin::TYPE> >
{
  $1 = PyList_Check($input) ? 1 : 0;
}

//-----------------------------------------------------------------------------
// The std::vector<shared_ptr<Type> > typemap
//-----------------------------------------------------------------------------
%typemap (in) CONST_VECTOR std::vector<SWIG_SHARED_PTR_QNAMESPACE::
              shared_ptr<CONST dolfin::TYPE> > (
std::vector<SWIG_SHARED_PTR_QNAMESPACE::shared_ptr<CONST dolfin::TYPE> > tmp_vec,
SWIG_SHARED_PTR_QNAMESPACE::shared_ptr<dolfin::TYPE> tempshared,
dolfin::TYPE * arg)
{
  // IN_TYPEMAP_STD_VECTOR_OF_POINTERS(TYPE, CONST, CONST_VECTOR), shared_ptr version
  if (PyList_Check($input))
  {
    int size = PyList_Size($input);
    int res = 0;
    PyObject * py_item = 0;
    void * itemp = 0;
    int newmem = 0;
    tmp_vec.reserve(size);
    for (int i = 0; i < size; i++)
    {
      newmem = 0;
      py_item = PyList_GetItem($input, i);
      res = SWIG_ConvertPtrAndOwn(py_item, &itemp, $descriptor(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::TYPE > *), 0, &newmem);
      if (SWIG_IsOK(res))
      {
	if (itemp)
	{
	  tempshared = *(reinterpret_cast<SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::TYPE> *>(itemp));
	  tmp_vec.push_back(tempshared);
	}
	if (newmem & SWIG_CAST_NEW_MEMORY)
	  delete reinterpret_cast<SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::TYPE> *>(itemp);
      }
      else
      {
        SWIG_exception(SWIG_TypeError, "expected a list of shared_ptr<TYPE> (Bad conversion)");
      }
    }
    $1 = tmp_vec;
  }
  else
  {
    SWIG_exception(SWIG_TypeError, "list of TYPE expected");
  }
}

//-----------------------------------------------------------------------------
// The std::vector<shared_ptr<Type> > typecheck
//-----------------------------------------------------------------------------
%typecheck(SWIG_TYPECHECK_POINTER) CONST_VECTOR std::vector<SWIG_SHARED_PTR_QNAMESPACE::shared_ptr<CONST dolfin::TYPE> >
{
  $1 = PyList_Check($input) ? 1 : 0;
}

//-----------------------------------------------------------------------------
// Out typemap of std::vector<shared_ptr<Type> >
//-----------------------------------------------------------------------------
%typemap (out) std::vector<SWIG_SHARED_PTR_QNAMESPACE::
               shared_ptr<CONST dolfin::TYPE> > (
SWIG_SHARED_PTR_QNAMESPACE::shared_ptr<CONST dolfin::TYPE> tempshared,
PyObject* ret_list,
PyObject* list_item)
{
  // OUT_TYPEMAP_STD_VECTOR_OF_POINTERS(TYPE, CONST, CONST_VECTOR), shared_ptr version
  int size = (&$1)->size();
  ret_list = PyList_New(size);

  // Iterate over the vector and fill list
  for (int i=0; i<size; i++)
  {
    // Grab the item
    tempshared = (&$1)->operator[](i);

    // Create a new ptr while increasing the reference.
    // NOTE: Const cast because SWIG does not know how to handle non
    // NOTE: const shared_ptr types
    SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::TYPE >* smartresult = tempshared ? new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::TYPE >(boost::const_pointer_cast<dolfin::TYPE>(tempshared)) : 0;
    list_item = SWIG_NewPointerObj(SWIG_as_voidptr(smartresult), $descriptor(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::TYPE > *), SWIG_POINTER_OWN);
    PyList_SET_ITEM(ret_list, i, list_item);
  }

  // Assign the result
  $result = ret_list;
}

%enddef

//-----------------------------------------------------------------------------
// Macro for defining an in typemap for a const std::vector& of primitives
// The typemaps takes a NumPy array of that primitive
//
// TYPE       : The primitive type
// TYPE_UPPER : The SWIG specific name of the type used in the array type checks
//              values SWIG use: INT32 for integer, DOUBLE for double aso.
// ARG_NAME   : The name of the argument that will be maped as an 'argout' argument
// NUMPY_TYPE : The type of the NumPy array that will be returned
// TYPE_NAME  : The name of the pointer type, 'double' for 'double', 'uint' for
//              'dolfin::uint'
// DESCR      : The char descriptor of the NumPy type
//-----------------------------------------------------------------------------
%define IN_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(TYPE, TYPE_UPPER, ARG_NAME, \
					    NUMPY_TYPE, TYPE_NAME, DESCR)

// The typecheck
%typecheck(SWIG_TYPECHECK_ ## TYPE_UPPER ## _ARRAY)  \
const std::vector<TYPE>&  ARG_NAME
{
  $1 = PyArray_Check($input) ? 1 : 0;
}

// The typemap
%typemap(in) const std::vector<TYPE>& ARG_NAME (std::vector<TYPE> temp)
{
  // IN_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(TYPE, TYPE_UPPER, ARG_NAME,
  //                                     NUMPY_TYPE, TYPE_NAME, DESCR)
  {
    if (PyArray_Check($input))
    {
      PyArrayObject *xa = reinterpret_cast<PyArrayObject*>($input);
      if ( PyArray_TYPE(xa) == NUMPY_TYPE )
      {
        const unsigned int size = PyArray_DIM(xa, 0);
        temp.resize(size);
        TYPE* array = static_cast<TYPE*>(PyArray_DATA(xa));
        if (PyArray_ISCONTIGUOUS(xa))
          std::copy(array, array + size, temp.begin());
        else
        {
          const npy_intp strides = PyArray_STRIDE(xa, 0)/sizeof(TYPE);
          for (int i=0; i<size; i++)
            temp[i] = array[i*strides];
        }
        $1 = &temp;
      }
      else
      {
        SWIG_exception(SWIG_TypeError, "numpy array of 'TYPE_NAME' expected."\
          " Make sure that the numpy array use dtype='DESCR'.");
      }
    }
    else
    {
      SWIG_exception(SWIG_TypeError, "numpy array of 'TYPE_NAME' expected. "\
		     "Make sure that the numpy array use dtype='DESCR'.");
    }
  }
}

%enddef

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
%define ARGOUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(TYPE, TYPE_UPPER, ARG_NAME, NUMPY_TYPE)

//-----------------------------------------------------------------------------
// In typemap removing the argument from the expected in list
//-----------------------------------------------------------------------------
%typemap (in,numinputs=0) std::vector<TYPE>& ARG_NAME (std::vector<TYPE> vec_temp)
{
  // ARGOUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(TYPE, TYPE_UPPER, ARG_NAME, NUMPY_TYPE)
  $1 = &vec_temp;
}

//-----------------------------------------------------------------------------
// Argout typemap, returning a NumPy array for the std::vector<TYPE>
//-----------------------------------------------------------------------------
%typemap(argout) std::vector<TYPE>& ARG_NAME
{
  PyObject* o0 = 0;
  PyObject* o1 = 0;
  PyObject* o2 = 0;
  npy_intp size = $1->size();
  PyArrayObject *ret = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size, NUMPY_TYPE));
  TYPE* data = static_cast<TYPE*>(PyArray_DATA(ret));
  for (int i = 0; i < size; ++i)
    data[i] = (*$1)[i];
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

//-----------------------------------------------------------------------------
// Macro for defining an in typemap for a std::vector of primitives passed by value
//
// TYPE       : The primitive type
// TYPE_UPPER : The SWIG specific name of the type used in the array type checks
//              values SWIG use: INT32 for integer, DOUBLE for double aso.
// ARG_NAME   : The name of the argument that will be maped as an 'argout' argument
// TYPE_NAME  : The name of the pointer type, 'double' for 'double', 'uint' for
//              'dolfin::uint'
// SEQ_LENGTH : An optional sequence length argument. If set to a negative number
//              will no length check be made
//-----------------------------------------------------------------------------
%define PY_SEQUENCE_OF_SCALARS_TO_VECTOR_OF_PRIMITIVES(TYPE, TYPE_UPPER, ARG_NAME,
						       TYPE_NAME, SEQ_LENGTH)

%typecheck(SWIG_TYPECHECK_ ## TYPE_UPPER ## _ARRAY) std::vector<TYPE> ARG_NAME
{
  $1 = PySequence_Check($input) ? 1 : 0;
}

%typemap (in, fragment=Py_convert_frag(TYPE_NAME)) std::vector<TYPE> ARG_NAME
(std::vector<TYPE> tmp_vec, PyObject* item, TYPE value, dolfin::uint i)
{
  // PY_SEQUENCE_OF_SCALARS_TO_VECTOR_OF_PRIMITIVES(TYPE, TYPE_UPPER,
  //                                    ARG_NAME, TYPE_NAME, SEQ_LENGTH)

  // A first sequence test
  if (!PySequence_Check($input))
    SWIG_exception(SWIG_TypeError, "expected a sequence for argument $argnum");

  // Get sequence length
  Py_ssize_t pyseq_length = PySequence_Size($input);
  if (SEQ_LENGTH >= 0 && pyseq_length > SEQ_LENGTH)
    SWIG_exception(SWIG_TypeError, "expected a sequence with length "	\
		   "SEQ_LENGTH for argument $argnum");

  tmp_vec.reserve(pyseq_length);
  for (i = 0; i < pyseq_length; i++)
  {
    item = PySequence_GetItem($input, i);

    if(!SWIG_IsOK(Py_convert_ ## TYPE_NAME(item, value)))
    {
      SWIG_exception(SWIG_TypeError, "expected items of sequence to be of type "\
		     "\"TYPE_NAME\" in argument $argnum");
    }
    tmp_vec.push_back(value);
    Py_DECREF(item);
  }
  $1 = tmp_vec;
}
%enddef

//-----------------------------------------------------------------------------
// Macro for out typemaps of primitives of const std::vector<TYPE>& It returns
// readonly NumPy array
//
// TYPE      : The primitive type
// TYPE_NAME : The name of the pointer type, 'double' for 'double', 'uint' for
//             'dolfin::uint'
//-----------------------------------------------------------------------------
%define READONLY_OUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(TYPE, TYPE_NAME)

%typemap(out, fragment=make_numpy_array_frag(1, TYPE_NAME)) const std::vector<TYPE>& {
  // READONLY_OUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(TYPE, TYPE_NAME)
  $result = %make_numpy_array(1, TYPE_NAME)($1->size(), &($1->operator[](0)), false);
}

%enddef

%define IN_TYPEMAP_STD_VECTOR_OF_SMALL_DOLFIN_TYPES(TYPE)
//-----------------------------------------------------------------------------
// Typemap for const std::vector<dolfin::TYPE>& used in IntersectionOperator
// Expects a list of Points
//-----------------------------------------------------------------------------
%typecheck(SWIG_TYPECHECK_POINTER) const std::vector<dolfin::TYPE>&
{
  $1 = PyList_Check($input) ? 1 : 0;
}

%typemap (in) const std::vector<dolfin::TYPE>& (std::vector<dolfin::TYPE> tmp_vec)
{
  // IN_TYPEMAP_STD_VECTOR_OF_SMALL_DOLFIN_TYPES, TYPE
  // A first sequence test
  if (!PyList_Check($input))
    SWIG_exception(SWIG_TypeError, "expected a list of TYPE for argument $argnum");

  int size = PyList_Size($input);
  int res = 0;
  PyObject * py_item = 0;
  void * itemp = 0;
  tmp_vec.reserve(size);
  for (int i = 0; i < size; i++)
  {
    py_item = PyList_GetItem($input,i);
    res = SWIG_ConvertPtr(py_item, &itemp, $descriptor(dolfin::TYPE*), 0);
    if (SWIG_IsOK(res))
    {
      tmp_vec.push_back(*reinterpret_cast<dolfin::TYPE *>(itemp));
    }
    else
    {
      SWIG_exception(SWIG_TypeError, "expected a list of TYPE for argument $argnum, (Bad conversion)");
    }
  }
  $1 = &tmp_vec;
}
%enddef

//-----------------------------------------------------------------------------
// Run the different macros and instantiate the typemaps
//-----------------------------------------------------------------------------
TYPEMAPS_STD_VECTOR_OF_POINTERS(DirichletBC)
TYPEMAPS_STD_VECTOR_OF_POINTERS(BoundaryCondition)
TYPEMAPS_STD_VECTOR_OF_POINTERS(GenericFunction)
TYPEMAPS_STD_VECTOR_OF_POINTERS(FunctionSpace)
TYPEMAPS_STD_VECTOR_OF_POINTERS(Parameters)

ARGOUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(dolfin::uint, INT32, cells, NPY_INT)
ARGOUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(dolfin::uint, INT32, columns, NPY_INT)
ARGOUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(double, DOUBLE, values, NPY_DOUBLE)

IN_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(double, DOUBLE, values, NPY_DOUBLE, double, d)
IN_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(dolfin::uint, INT32, columns, NPY_UINT, uint, I)

PY_SEQUENCE_OF_SCALARS_TO_VECTOR_OF_PRIMITIVES(dolfin::uint, INT32, coloring_type, uint, -1)
PY_SEQUENCE_OF_SCALARS_TO_VECTOR_OF_PRIMITIVES(dolfin::uint, INT32, value_shape, uint, -1)
PY_SEQUENCE_OF_SCALARS_TO_VECTOR_OF_PRIMITIVES(unsigned int, INT32, coloring_type, uint, -1)
PY_SEQUENCE_OF_SCALARS_TO_VECTOR_OF_PRIMITIVES(unsigned int, INT32, value_shape, uint, -1)
PY_SEQUENCE_OF_SCALARS_TO_VECTOR_OF_PRIMITIVES(double, DOUBLE, values, double, -1)

READONLY_OUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(double, double)
READONLY_OUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(int, int)
READONLY_OUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(unsigned int, uint)
READONLY_OUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(dolfin::uint, uint)

IN_TYPEMAP_STD_VECTOR_OF_SMALL_DOLFIN_TYPES(Point)
IN_TYPEMAP_STD_VECTOR_OF_SMALL_DOLFIN_TYPES(MeshEntity)
