/* -*- C -*- */
// Copyright (C) 2007-2009 Anders logg
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007-2009.
// Modified by Garth N. Wells, 2007.
// Modified by Johan Hake, 2008-2009.
//
// First added:  2006-04-16
// Last changed: 2009-09-07

//=============================================================================
// General typemaps for PyDOLFIN
//=============================================================================

//-----------------------------------------------------------------------------
// A hack to get around incompatabilities with PyInt_Check and numpy int 
// types in python 2.6
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// The typecheck
//-----------------------------------------------------------------------------
%typecheck(SWIG_TYPECHECK_INTEGER) dolfin::uint
{
    $1 = PyInt_Check($input) || PyType_IsSubtype($input->ob_type, &PyInt_Type) ? 1 : 0;
}

//-----------------------------------------------------------------------------
// The typemap
//-----------------------------------------------------------------------------
%typemap(in) dolfin::uint
{
  if (PyInt_Check($input) || PyType_IsSubtype($input->ob_type, &PyInt_Type))
  {
    long tmp = PyInt_AsLong($input);
    if (tmp>=0)
      $1 = static_cast<dolfin::uint>(tmp);
    else
      SWIG_exception(SWIG_TypeError, "positive 'int' expected");
  }
  else
    SWIG_exception(SWIG_TypeError, "positive 'int' expected");
}

//-----------------------------------------------------------------------------
// Apply the builtin out-typemap for int to dolfin::uint
//-----------------------------------------------------------------------------
%typemap(out) dolfin::uint = int;

//-----------------------------------------------------------------------------
// Out typemap for std::pair<uint,uint>
//-----------------------------------------------------------------------------
%typemap(out) std::pair<dolfin::uint,dolfin::uint>
{
  $result = Py_BuildValue("ii",$1.first,$1.second);
}
