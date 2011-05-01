/* -*- C -*- */
// Copyright (C) 2009 Johan Hake
// Licensed under the GNU LGPL Version 2.1.
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

