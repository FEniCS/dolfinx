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
// Last changed: 2011-02-09

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

%wrapper%{
// A check for float and converter for double
SWIGINTERNINLINE bool Py_float_convert(PyObject* in, double& value)
{
  return SWIG_AsVal_double(in, &value);
}

// A check for int and converter for int
SWIGINTERNINLINE bool Py_int_convert(PyObject* in, int& value)
{
  if (!PyInteger_Check(in))
    return false;
  long tmp = static_cast<long>(PyInt_AsLong(in));
  value = static_cast<dolfin::uint>(tmp);
  return true;
  //return SWIG_AsVal_int(in, &value);
}

// A check for int and converter for uint
SWIGINTERNINLINE bool Py_uint_convert(PyObject* in, dolfin::uint& value)
{
  if (!PyInteger_Check(in))
    return false;
  long tmp = static_cast<long>(PyInt_AsLong(in));
  if (tmp<=0)
    return false;
  value = static_cast<dolfin::uint>(tmp);
  return true;
  //return SWIG_AsVal_unsigned_SS_int(in, &value);
}
%}
//-----------------------------------------------------------------------------
// Apply the builtin out-typemap for int to dolfin::uint
//-----------------------------------------------------------------------------
%typemap(out) dolfin::uint
{
  $result = SWIG_From_unsigned_SS_int($1);
}

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
