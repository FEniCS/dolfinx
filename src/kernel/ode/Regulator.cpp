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
void Regulator::update(real k)
{
  real k0 = this->k;
  real k1 = k;

  this->k = 2.0 * k0 * k1 / (k0 + k1);
}
//-----------------------------------------------------------------------------
real Regulator::timestep() const
{
  return k;
}
//-----------------------------------------------------------------------------
