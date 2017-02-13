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
// Extend Point with Python sequence interface
//-----------------------------------------------------------------------------
%extend dolfin::Point {
  // Wrap operator[] (now without bound checks)
  double _getitem(std::size_t i)
  {
    return (*self)[i];
  }

  void _setitem(std::size_t i, double val)
  {
    (*self)[i] = val;
  }

  // Implement type and bound checks assuming Point is 3D
  %pythoncode %{
    def __len__(self):
      return 3

    def __getitem__(self, i):
      "Get i-th coordinate. Only accept integers, not slices."
      return self._getitem(self._check_index(i))

    def __setitem__(self, i, value):
      "Set i-th coordinate. Only accept integers, not slices."
      self._setitem(self._check_index(i), value)

    from numpy import uintp

    def _check_index(self, i):
      "Check index is convertible to uintp and in range"
      # Accept only integral types, not slices
      i = self.uintp(i)

      # Range check
      if i > 2:
        raise IndexError("Dimension of Point is always 3")

      # Return size_t index
      return i
  %}
}
