/* -*- C -*- */
// Copyright (C) 2009 Johan Hake
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-10-07
// Last changed: 2009-10-07

//=============================================================================
// SWIG directives for the DOLFIN nls kernel module (pre)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
//=============================================================================

//-----------------------------------------------------------------------------
// Add director classes
//-----------------------------------------------------------------------------
%feature("director") dolfin::NonlinearProblem;
%feature("director") dolfin::NonlinearProblemTest;
