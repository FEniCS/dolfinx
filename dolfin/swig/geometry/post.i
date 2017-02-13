/* -*- C -*- */
// Copyright (C) 2006-2009 Anders Logg
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
// Modified by Johan Jansson 2006-2007
// Modified by Ola Skavhaug 2006-2007
// Modified by Garth Wells 2007-2010
// Modified by Johan Hake 2008-2009
//
// First added:  2006-09-20
// Last changed: 2011-03-11

//=============================================================================
// SWIG directives for the DOLFIN Geometry kernel module (post)
//
// The directives in this file are applied _after_ the header files of the
// modules has been loaded.
//=============================================================================

//-----------------------------------------------------------------------------
// Extend Point interface with Python selectors
//-----------------------------------------------------------------------------
%feature("docstring") dolfin::Point::__getitem__ "Missing docstring";
%feature("docstring") dolfin::Point::__setitem__ "Missing docstring";

//-----------------------------------------------------------------------------
// Macro for exception handler which turn C++ exception into SWIG exception
//
// NAME          : Name of the function to be handled
// CPP_EXC_TYPE  : C++ type of exception to be handled
// SWIG_EXC_CODE : SWIG code of exception as defined in exception.i
//-----------------------------------------------------------------------------
%define CPP_EXC_TO_SWIG(NAME, CPP_EXC_TYPE, SWIG_EXC_CODE)
%include "exception.i"
%exception NAME
{
  try
  {
    $action
  }
  catch (CPP_EXC_TYPE &e)
  {
    SWIG_exception(SWIG_EXC_CODE, e.what());
  }
}
%enddef

CPP_EXC_TO_SWIG(dolfin::Point::__getitem__, std::range_error, SWIG_IndexError)
CPP_EXC_TO_SWIG(dolfin::Point::__setitem__, std::range_error, SWIG_IndexError)

%extend dolfin::Point {
  double __len__()
  {
    return 3;
  }
  double __getitem__(int i)
  {
    if (i > 2)
      throw std::range_error("Dimension of Point is always 3.");
    return (*self)[i];
  }
  void __setitem__(int i, double val)
  {
    if (i > 2)
      throw std::range_error("Dimension of Point is always 3.");
    (*self)[i] = val;
  }
}
