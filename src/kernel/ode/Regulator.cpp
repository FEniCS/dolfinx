// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

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
void Regulator::update(real k, real kmax)
{
  // Old time step
  real k0 = this->k;

  // Desired time step
  real k1 = k;

  // Check kmax
  if ( k1 > kmax )
    k1 = kmax;

  // Compute new time step
  this->k = 2.0 * k0 * k1 / (k0 + k1);
}
//-----------------------------------------------------------------------------
real Regulator::timestep() const
{
  return k;
}
//-----------------------------------------------------------------------------
