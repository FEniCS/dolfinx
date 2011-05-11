/* -*- C -*- */
// Copyright (C) 2007-2009 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Ola Skavhaug, 2007-2009.
// Modified by Garth N. Wells, 2007.
// Modified by Johan Hake, 2008-2009.
//
// First added:  2006-04-16
// Last changed: 2011-05-02

//=============================================================================
// General typemaps for PyDOLFIN
//=============================================================================

// Ensure typefragments
%ensure_type_fragments(double)
%ensure_type_fragments(unsigned int)

//-----------------------------------------------------------------------------
// A home brewed type check for checking integers
// Needed due to problems with PyInt_Check from python 2.6 and NumPy
//-----------------------------------------------------------------------------
%fragment("PyInteger_Check", "header") {
  SWIGINTERNINLINE bool PyInteger_Check(PyObject* in)
  {
    return  PyInt_Check(in) || (PyArray_CheckScalar(in) &&
  			      PyArray_IsScalar(in,Integer));
  }
}

#define Py_convert_frag(Type) "Py_convert_" {Type}

%fragment("Py_convert_double", "header") {
  // A check for float and converter for double
  SWIGINTERNINLINE bool Py_convert_double(PyObject* in, double& value)
  {
    return SWIG_AsVal(double)(in, &value);
  }
}

%fragment("Py_convert_int", "header", fragment="PyInteger_Check") {
  // A check for int and converter for int
  SWIGINTERNINLINE bool Py_convert_int(PyObject* in, int& value)
  {
    if (!PyInteger_Check(in))
      return false;
    long tmp = static_cast<long>(PyInt_AsLong(in));
    value = static_cast<dolfin::uint>(tmp);
    return true;
  }
}

%fragment("Py_convert_uint", "header", fragment="PyInteger_Check") {
  // A check for int and converter to uint
  SWIGINTERNINLINE bool Py_convert_uint(PyObject* in, dolfin::uint& value)
  {
    if (!PyInteger_Check(in))
      return false;
    long tmp = static_cast<long>(PyInt_AsLong(in));
    if (tmp<=0)
      return false;
    value = static_cast<dolfin::uint>(tmp);
    return true;
  }
}

//-----------------------------------------------------------------------------
// Apply the builtin out-typemap for int to dolfin::uint
//-----------------------------------------------------------------------------
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

%typemap(in, fragment=SWIG_AsVal_frag(double)) dolfin::real
{
  $1 = dolfin::to_real(PyFloat_AsDouble($input));
}

%typemap(out, fragment=SWIG_From_frag(double)) dolfin::real
{
  $result = SWIG_From(double)(dolfin::to_double($1));
}

//-----------------------------------------------------------------------------
// Typemaps for dolfin::uint and int
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// The typecheck (unsigned int)
//-----------------------------------------------------------------------------
%typecheck(SWIG_TYPECHECK_INTEGER) unsigned int
{
  $1 = PyInteger_Check($input) ? 1 : 0;
}

//-----------------------------------------------------------------------------
// The typemap (unsigned int)
//-----------------------------------------------------------------------------
%typemap(in, fragment="PyInteger_Check") unsigned int
{
  if (PyInteger_Check($input))
  {
    long tmp = static_cast<long>(PyInt_AsLong($input));
    if (tmp>=0)
      $1 = static_cast<unsigned int>(tmp);
    else
      SWIG_exception(SWIG_TypeError, "expected positive 'int' for argument $argnum");
  }
  else
    SWIG_exception(SWIG_TypeError, "expected positive 'int' for argument $argnum");
}

%typemap(out, fragment=SWIG_From_frag(unsigned int)) unsigned int
{
  // Typemap unsigned int
  $result = SWIG_From(unsigned int)($1);
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
%typemap(in, fragment="PyInteger_Check") int
{
  if (PyInteger_Check($input))
  {
    long tmp = static_cast<long>(PyInt_AsLong($input));
    $1 = static_cast<int>(tmp);
  }
  else
    SWIG_exception(SWIG_TypeError, "expected 'int' for argument $argnum");
}
