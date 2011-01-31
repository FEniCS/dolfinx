/* -*- C -*- */
// Copyright (C) 2005-2006 Johan Hake
// Licensed under the GNU LGPL Version 2.1.
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
