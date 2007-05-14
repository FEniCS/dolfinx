// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2005-12-19
// Last changed: 2006-03-27

#include <limits>
#include <dolfin/dolfin_log.h>
#include <dolfin/ParameterSystem.h>

// Initialize the global parameter database
dolfin::ParameterSystem dolfin::ParameterSystem::parameters;

using namespace dolfin;

//-----------------------------------------------------------------------------
ParameterSystem::ParameterSystem() : ParameterList()
{
#include <dolfin/DefaultParameters.h>
}
//-----------------------------------------------------------------------------
