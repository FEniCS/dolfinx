// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <iostream>
#include <dolfin/dolfin_log.h>
#include <dolfin/Adaptivity.h>
#include <dolfin/RHS.h>
#include <dolfin/Solution.h>
#include <dolfin/FixedPointIteration.h>
#include <dolfin/TimeSlab.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
TimeSlab::TimeSlab(real t0, real t1)
{
  this->t0 = t0;
  this->t1 = t1;
  reached_endtime = false;
}
//-----------------------------------------------------------------------------
TimeSlab::~TimeSlab()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool TimeSlab::within(real t) const
{
  // Check if t is in the interval (t0,t1] = [t0 + eps, t1 + eps].
  // We need to make sure that we include the end-point. Otherwise
  // a round-off error may cause the next interval to be chosen,
  // which is not what we want, at least not for dG.

  // FIXME: Is this necessary?
  //t -= DOLFIN_EPS;

  //return (t0 <= t) && (t <= t1);
  return (t0 < t) && (t <= t1);
}
//-----------------------------------------------------------------------------
bool TimeSlab::finished() const
{
  return reached_endtime;
}
//-----------------------------------------------------------------------------
real TimeSlab::starttime() const
{
  return t0;
}
//-----------------------------------------------------------------------------
real TimeSlab::endtime() const
{
  return t1;
}
//-----------------------------------------------------------------------------
real TimeSlab::length() const
{
  return t1 - t0;
}
//-----------------------------------------------------------------------------
void TimeSlab::setsize(real K, const Adaptivity& adaptivity)
{
  // Make sure that we don't go beyond t1

  if ( K > adaptivity.threshold() * (t1 - t0) )
  {
    K = t1 - t0;
    t1 = t0 + K;
    reached_endtime = true;
  }
  else
    t1 = t0 + K;
}
//-----------------------------------------------------------------------------
real TimeSlab::updateElements(FixedPointIteration& fixpoint)
{
  real dmax = 0.0;

  // Update elements
  for (unsigned int i = 0; i < elements.size(); i++)
  {
    // Get the element
    Element* element = elements[i];
    dolfin_assert(element);

    // Update element
    dmax = std::max(dmax, fixpoint.update(*element));
  }

  return dmax;
}
//-----------------------------------------------------------------------------
void TimeSlab::resetElements(FixedPointIteration& fixpoint)
{
  // Reset elements
  for (unsigned int i = 0; i < elements.size(); i++)
  {
    // Get the element
    Element* element = elements[i];
    dolfin_assert(element);
    
    // Reset element
    fixpoint.reset(*element);
  }
}
//-----------------------------------------------------------------------------
real TimeSlab::computeMaxRdElements(Solution& u, RHS& f)
{
  real maxrd = 0.0;

  // Compute maximum discrete residual
  for (unsigned int i = 0; i < elements.size(); i++)
  {
    // Get the element
    Element* element = elements[i];
    dolfin_assert(element);

    // Compute discrete residual
    maxrd = std::max(maxrd, fabs(element->computeDiscreteResidual(f)));
    
    //cout << "  r[" << element->index() << "] = "
    //	   << fabs(element->computeDiscreteResidual(f)) << endl;

  }

  return maxrd;
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<<(LogStream& stream, const TimeSlab& timeslab)
{
  stream << "[ TimeSlab of length " << timeslab.length()
	 << " between t0 = " << timeslab.starttime()
	 << " and t1 = " << timeslab.endtime() << " ]";

  return stream;
}
//-----------------------------------------------------------------------------
