/* -*- C -*- */
// Copyright (C) 2007-2009 Anders logg
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007-2009.
// Modified by Garth N. Wells, 2007.
// Modified by Johan Hake, 2008-2009.
//
// First added:  2006-04-16
// Last changed: 2009-09-13

//=============================================================================
// General typemaps for PyDOLFIN
//=============================================================================

//-----------------------------------------------------------------------------
// A home brewed type check for checking integers 
// Needed due to problems with PyInt_Check from python 2.6 and NumPy
//-----------------------------------------------------------------------------
%{
bool PyInteger_Check(PyObject* in)
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
// Out typemap for std::pair<uint,uint>
//-----------------------------------------------------------------------------
%typemap(out) std::pair<dolfin::uint,dolfin::uint>
{
  $result = Py_BuildValue("ii",$1.first,$1.second);
}
