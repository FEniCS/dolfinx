/* -*- C -*- */
// Copyright (C) 2007-2009 Anders logg
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007-2009.
// Modified by Garth N. Wells, 2007.
// Modified by Johan Hake, 2008-2009.
//
// First added:  2006-04-16
// Last changed: 2009-10-07

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
// In typemap for std::pair<TYPE,TYPE>
//-----------------------------------------------------------------------------
%typemap(typecheck,precedence=SWIG_TYPECHECK_DOUBLE_ARRAY) const std::pair<const dolfin::Function*,double>
{
  SWIG_exception(SWIG_TypeError, "pair typemap not impl (a).");
}

%typemap(typecheck,precedence=SWIG_TYPECHECK_DOUBLE_ARRAY) const std::pair<dolfin::Function*,double>
{
  SWIG_exception(SWIG_TypeError, "pair typemap not impl (a).");
}

%typemap(typecheck,precedence=SWIG_TYPECHECK_DOUBLE_ARRAY) std::pair<const dolfin::Function*,double>
{
  SWIG_exception(SWIG_TypeError, "pair typemap not impl (b).");
}

%typemap(typecheck,precedence=SWIG_TYPECHECK_DOUBLE_ARRAY) std::pair<dolfin::Function*,double>
{
  SWIG_exception(SWIG_TypeError, "pair typemap not impl (b).");
}


%typemap(in) const std::pair<const dolfin::Function*,double> (std::pair<const dolfin::Function*,double> tmp_pair)
{
  // Check that we don't have a list
  if (PyList_Check($input))
    SWIG_exception(SWIG_TypeError, "A tuple is expected (list received)..");

  // Check that we have a tuple
  if (!PyTuple_Check($input))
   SWIG_exception(SWIG_TypeError, "Tuple expected.");

  // Check tuple length
  int size = PyTuple_Size($input);
  if (size != 2)
    SWIG_exception(SWIG_TypeError, "Tuple of length two expected.");

  // Get pointers to function and time
  PyObject* py_function = PyTuple_GetItem($input, 0);
  PyObject* py_time     = PyTuple_GetItem($input, 1);

  // Check that we have a float
  if (!PyFloat_Check(py_time))
    SWIG_exception(SWIG_TypeError, "Float expected for time.");

  // Get time
  double time = PyFloat_AsDouble(py_time);

  //const dolfin::Function* test_function = reinterpret_cast<const dolfin::Function*>(py_function);
  //double test_d = *reinterpret_cast<double*>(py_time);

  std::cout << "About to test function " << std::endl;
  std::cout << "Testing function " << time << std::endl;


  tmp_pair.first  = reinterpret_cast<const dolfin::Function*>(py_function);
  tmp_pair.second = time;


  $1 = tmp_pair;
  SWIG_exception(SWIG_TypeError, "const std::pair<const dolfin::Function*,double> typemap not implemented.");
}

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
