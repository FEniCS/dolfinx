namespace std
{
  template <class T> class set 
  {
  };
}


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
