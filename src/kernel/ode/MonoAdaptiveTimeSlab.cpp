// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/MonoAdaptiveTimeSlab.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MonoAdaptiveTimeSlab::MonoAdaptiveTimeSlab(ODE& ode) : NewTimeSlab(ode)
{
  

}
//-----------------------------------------------------------------------------
MonoAdaptiveTimeSlab::~MonoAdaptiveTimeSlab()
{


}
//-----------------------------------------------------------------------------
real MonoAdaptiveTimeSlab:: build(real a, real b)
{


  return 0.0;
}
//-----------------------------------------------------------------------------
void MonoAdaptiveTimeSlab::solve()
{


}
//-----------------------------------------------------------------------------
void MonoAdaptiveTimeSlab::shift()
{


}
//-----------------------------------------------------------------------------
void MonoAdaptiveTimeSlab::sample(real t)
{

}
//-----------------------------------------------------------------------------
real MonoAdaptiveTimeSlab::usample(uint i, real t)
{


  return 0.0;
}
//-----------------------------------------------------------------------------
real MonoAdaptiveTimeSlab::ksample(uint i, real t)
{


  return 0.0;
}
//-----------------------------------------------------------------------------
real MonoAdaptiveTimeSlab::rsample(uint i, real t)
{

  return 0.0;
}
//-----------------------------------------------------------------------------
void MonoAdaptiveTimeSlab::disp() const
{


}
//-----------------------------------------------------------------------------
