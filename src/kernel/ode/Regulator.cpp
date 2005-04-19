// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Jansson, 2003.

#include <dolfin/dolfin_log.h>
#include <dolfin/Regulator.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Regulator::Regulator()
{
  k = 0.0;
}
//-----------------------------------------------------------------------------
Regulator::Regulator(real k)
{
  this->k = k;
}
//-----------------------------------------------------------------------------
Regulator::~Regulator()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Regulator::init(real k)
{
  this->k = k;
}
//-----------------------------------------------------------------------------
void Regulator::update(real k, real kmax, real w, bool kfixed)
{
  // Check if we should use a fixed time step
  if ( !kfixed )
  {
    // Old time step
    const real k0 = this->k;
    
    // Compute new time step
    this->k = (1.0 + w) * k0 * k / (k0 + w*k);
  }

  // Check kmax
  if ( this->k > kmax )
    this->k = kmax;
}
//-----------------------------------------------------------------------------
void Regulator::update(real kmax)
{
  if ( k > kmax )
    k = kmax;
}
//-----------------------------------------------------------------------------
real Regulator::timestep() const
{
  return k;
}
//-----------------------------------------------------------------------------
