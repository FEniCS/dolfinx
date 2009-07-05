// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-07-02
// Last changed: 2009-07-02

#include "GlobalParameters.h"

using namespace dolfin;

/// The global parameter database
GlobalParameters dolfin::parameters;

//-----------------------------------------------------------------------------
GlobalParameters::GlobalParameters() : Parameters("dolfin")
{
  // Set default parameter values
  *static_cast<Parameters*>(this) = default_parameters();
}
//-----------------------------------------------------------------------------
GlobalParameters::~GlobalParameters()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
