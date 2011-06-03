/* -*- C -*- */
// Copyright (C) 2009 Johan Hake
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
// First added:  2007-10-22
// Last changed: 2010-02-02

// ===========================================================================
// SWIG directives for the DOLFIN fem kernel module (post)
//
// The directives in this file are applied _after_ the header files of the
// modules has been loaded.
// ===========================================================================

//-----------------------------------------------------------------------------
// Extend Function so f.function_space() return a dolfin.FunctionSpace
//-----------------------------------------------------------------------------
%extend dolfin::BoundaryCondition {
%pythoncode %{
def function_space(self):
    "Return the FunctionSpace"
    from dolfin.functions.functionspace import FunctionSpaceFromCpp
    return FunctionSpaceFromCpp(self._function_space())
%}
}

