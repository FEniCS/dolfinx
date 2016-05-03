/* -*- C -*- */
// Copyright (C) 2011 Marie E. Rognes
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
// Modified by Garth N. Wells, 2011.
//
// First added:  2011-02-23
// Last changed: 2011-07-05

// ===========================================================================
// SWIG directives for the DOLFIN adaptivity kernel module (pre)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
// ===========================================================================

//-----------------------------------------------------------------------------
// Instantiate Hierarchical ErrorControl template class
//-----------------------------------------------------------------------------
%template (HierarchicalErrorControl) dolfin::Hierarchical<dolfin::ErrorControl>;

//-----------------------------------------------------------------------------
// Remove silly warning from friend function
//-----------------------------------------------------------------------------
%warnfilter(503) ErrorControl;

//-----------------------------------------------------------------------------
// Ignore GoalFunctional entirely
//-----------------------------------------------------------------------------
%ignore dolfin::GoalFunctional;

//-----------------------------------------------------------------------------
// Wrap adapt return value with PyDOLFIN classes
//-----------------------------------------------------------------------------
%pythonappend dolfin::adapt %{
from dolfin import cpp
from dolfin.functions import Function, FunctionSpace
if isinstance(val, cpp.Function):
    return Function(val)
if isinstance(val, cpp.FunctionSpace):
    return FunctionSpace(val)
# NOTE: Mesh will possibly appear here when Mesh has special PyDOLFIN wrapper
#       subclassing ufl.Domain
return val
%}
