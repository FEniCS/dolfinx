/* -*- C -*- */
// Copyright (C) 2005-2006 Johan Hake
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
// First added:  2009-09-07
// Last changed: 2011-01-31

//=============================================================================
// SWIG directives for the DOLFIN ode kernel module (pre)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
//=============================================================================

//-----------------------------------------------------------------------------
// Make SWIG aware of std iterator type
// This to stop warnings from SWIG
// FIXME: This is not a solution to how we should wrapp ODESolution best.
//-----------------------------------------------------------------------------
namespace std
{
  template <class T0, class T1*> class iterator
  {
  };
}

%template () std::iterator<std::input_iterator_tag, dolfin::ODESolutionData*>;

//-----------------------------------------------------------------------------
// Ignore operator++ so SWIG stop complaining
//-----------------------------------------------------------------------------
%ignore dolfin::ODESolutionIterator::operator++;

//-----------------------------------------------------------------------------
// Add director classes
//-----------------------------------------------------------------------------
%feature("director") dolfin::ODE;
