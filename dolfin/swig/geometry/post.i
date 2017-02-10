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

%include "exception.i"

%exception dolfin::Point::__getitem__
{
  try
  {
    $action
  }
  catch (std::range_error &e)
  {
    SWIG_exception(SWIG_IndexError, "Index out of bounds");
  }
}

%extend dolfin::Point {
  double __len__() { return 3; }
  double __getitem__(int i) {
    if (i > 2)
      throw std::range_error("");
    return (*self)[i]; }
  void __setitem__(int i, double val) { (*self)[i] = val; }
}
