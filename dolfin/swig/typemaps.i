/* -*- C -*- */
// Copyright (C) 2007-2009 Anders logg
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007-2009.
// Modified by Garth N. Wells, 2007.
// Modified by Johan Hake, 2008-2009.
//
// First added:  2006-04-16
// Last changed: 2010-06-02

//=============================================================================
// General typemaps for PyDOLFIN
//=============================================================================

//-----------------------------------------------------------------------------
// A home brewed type check for checking integers
// Needed due to problems with PyInt_Check from python 2.6 and NumPy
//-----------------------------------------------------------------------------
%{
SWIGINTERNINLINE bool PyInteger_Check(PyObject* in)
{
  return  PyInt_Check(in) || (PyArray_CheckScalar(in) &&
			      PyArray_IsScalar(in,Integer));
}
%}

//-----------------------------------------------------------------------------
// Apply the builtin out-typemap for int to dolfin::uint
//-----------------------------------------------------------------------------
%typemap(out) dolfin::uint = int;

//-----------------------------------------------------------------------------
// Typemaps for dolfin::real
// We do not pass any high precision values here. This might change in future
// FIXME: We need to find out what to do with Parameters of real. Now they are
// treated the same as double, and need to have a different typecheck value than
// DOUBLE 90!= 95. However this will render real parameters unusefull if we do
// not pick something else thatn PyFloat_Check in the typecheck.
//-----------------------------------------------------------------------------
%typecheck(95) dolfin::real
{
  // When implementing high precision type, check for that here.
  $1 = PyFloat_Check($input) ? 1 : 0;
}

%typemap(in) dolfin::real
{
  $1 = dolfin::to_real(PyFloat_AsDouble($input));
}

%typemap(out) dolfin::real
{
  $result = SWIG_From_double(dolfin::to_double($1));
}

//-----------------------------------------------------------------------------
// Typemaps for dolfin::uint and int
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// The typecheck (dolfin::uint)
//-----------------------------------------------------------------------------
%typecheck(SWIG_TYPECHECK_INTEGER) dolfin::uint
{
  $1 = PyInteger_Check($input) ? 1 : 0;
}

//-----------------------------------------------------------------------------
// The typemap (dolfin::uint)
//-----------------------------------------------------------------------------
%typemap(in) dolfin::uint
{
  if (PyInteger_Check($input))
  {
    long tmp = static_cast<long>(PyInt_AsLong($input));
    if (tmp>=0)
      $1 = static_cast<dolfin::uint>(tmp);
    else
      SWIG_exception(SWIG_TypeError, "expected positive 'int' for argument $argnum");
  }
  else
    SWIG_exception(SWIG_TypeError, "expected positive 'int' for argument $argnum");
}

//-----------------------------------------------------------------------------
// The typecheck (int)
//-----------------------------------------------------------------------------
%typecheck(SWIG_TYPECHECK_INTEGER) int
{
  $1 =  PyInteger_Check($input) ? 1 : 0;
}

//-----------------------------------------------------------------------------
// The typemap (int)
//-----------------------------------------------------------------------------
%typemap(in) int
{
  if (PyInteger_Check($input))
  {
    long tmp = static_cast<long>(PyInt_AsLong($input));
    $1 = static_cast<int>(tmp);
  }
  else
    SWIG_exception(SWIG_TypeError, "expected 'int' for argument $argnum");
}


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
  if (SWIG_IsOK(res)) {
    tmp_pair.first = reinterpret_cast<dolfin::TYPE *>(itemp);
  }
  else{
    // If failed with normal pointer conversion then
    // try with shared_ptr conversion
    newmem = 0;
    res = SWIG_ConvertPtrAndOwn(py_first, &itemp, $descriptor(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::TYPE > *), 0, &newmem);
    if (SWIG_IsOK(res)){
      // If we need to release memory
      if (newmem & SWIG_CAST_NEW_MEMORY){
	tempshared = *reinterpret_cast< SWIG_SHARED_PTR_QNAMESPACE::shared_ptr<dolfin::TYPE> * >(itemp);
	delete reinterpret_cast< SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::TYPE > * >(itemp);
	arg = const_cast< dolfin::TYPE * >(tempshared.get());
      }
      else {
	arg = const_cast< dolfin::TYPE * >(reinterpret_cast< SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< dolfin::TYPE > * >(itemp)->get());
      }
      tmp_pair.first = arg;
    }
    else {
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
// Out typemap for std::pair<TYPE,TYPE>
//-----------------------------------------------------------------------------
%typemap(out) std::pair<dolfin::uint,dolfin::uint>
{
  $result = Py_BuildValue("ii",$1.first,$1.second);
}
%typemap(out) std::pair<dolfin::uint,bool>
{
  $result = Py_BuildValue("ib",$1.first,$1.second);
}
