/* -*- C -*- */
// Copyright (C) 2009 Johan Hake
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-05-10
// Last changed: 2009-09-02

//=============================================================================
// SWIG directives for the DOLFIN common kernel module (pre)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
//=============================================================================

//-----------------------------------------------------------------------------
// We cannot use rename as the extend directive in shared_ptr_classes.i will 
// confuse swig.
//-----------------------------------------------------------------------------
%ignore dolfin::Variable::str;
