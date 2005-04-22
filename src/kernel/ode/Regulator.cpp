// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Jansson, 2003.
// Modified by Anders Logg, 2005.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/Regulator.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Regulator::Regulator() : w(dolfin_get("time step conservation"))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Regulator::~Regulator()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real Regulator::regulate(real knew, real k0, real kmax, bool kfixed)
{
  real k = k0;
  
  // Check if we should use a fixed time step
  if ( !kfixed )
  {
    // Compute new time step
    k = (1.0 + w) * k0 * knew / (k0 + w*knew);
  }

  // Check kmax
  if ( k > kmax )
    k = kmax;

  return k;
}
//-----------------------------------------------------------------------------
