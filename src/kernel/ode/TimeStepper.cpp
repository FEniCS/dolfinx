// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Jansson 2003, 2004.

#include <cmath>
#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/timeinfo.h>
#include <dolfin/ODE.h>
#include <dolfin/Sample.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/SimpleTimeSlab.h>
#include <dolfin/RecursiveTimeSlab.h>
#include <dolfin/ElementGroupList.h>
#include <dolfin/ElementGroupIterator.h>
#include <dolfin/Element.h>
#include <dolfin/ElementIterator.h>
#include <dolfin/TimeStepper.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
TimeStepper::TimeStepper(ODE& ode, Function& function) :
  N(ode.size()), t(0), T(ode.endtime()), partition(N), adaptivity(ode),
  u(ode, function), f(ode, u), fixpoint(u, f, adaptivity), 
  file(u.label() + ".m"), p("Time-stepping"), _finished(false),
  save_solution(dolfin_get("save solution")),
  adaptive_samples(dolfin_get("adaptive samples")),
  no_samples(dolfin_get("number of samples")),
  sample_density(dolfin_get("sample density"))
				 
{
  //Do nothing
}
//-----------------------------------------------------------------------------
TimeStepper::~TimeStepper()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void TimeStepper::solve(ODE& ode, Function& function)
{
  // Create a TimeStepper object
  TimeStepper timeStepper(ode, function);

  // Do time stepping
  while ( !timeStepper.finished() )
    timeStepper.step();
}
//-----------------------------------------------------------------------------
real TimeStepper::step()
{
  if ( t == 0.0 )
  {
    dolfin_warning("ODE solver is EXPERIMENTAL.");

    // Start timing
    tic();

    // Create first time slab
    while ( !createFirstTimeSlab() );
  }
  else
    while ( !createGeneralTimeSlab() );

  return t;
}
//-----------------------------------------------------------------------------
bool TimeStepper::finished() const
{
  return _finished;
}
//-----------------------------------------------------------------------------
bool TimeStepper::createFirstTimeSlab()
{
  // Create the time slab
  SimpleTimeSlab timeslab(t, T, u, adaptivity);

  // Try to solve the system using fixed point iteration
  if ( !fixpoint.iterate(timeslab) )
  {
    stabilize(timeslab.length());
    u.reset();
    return false;
  }

  // Check if the residual is small enough if the time step is not fixed
  if ( !adaptivity.fixed() )
  {
    if ( !adaptivity.accept(timeslab, f) )
    {
      cout << "Residual is too large, creating a new time slab." << endl;
      adaptivity.shift(u, f);
      u.reset();
      return false;
    }
  }

  // Update time
  t = timeslab.endtime();
  
  // Save solution
  save(timeslab);
  
  // Prepare for next time slab
  shift();
  
  // Update progress
  p = t / T;

  // Check if we are done
  if ( timeslab.finished() )
  {
    _finished = true;
    p = 1.0;
    cout << "Solution computed in " << toc() << " seconds." << endl;
    fixpoint.report();
  }

  return true;
}
//-----------------------------------------------------------------------------
bool TimeStepper::createGeneralTimeSlab()
{
  // Create the time slab
  RecursiveTimeSlab timeslab(t, T, u, f, adaptivity, fixpoint, partition, 0);

  // Try to solve the system using fixed point iteration
  if ( !fixpoint.iterate(timeslab) )
  {
    stabilize(timeslab.length());
    u.reset();
    return false;
  }
  
  /*
  // Check if the residual is small enough
  if ( !adaptivity.accept(timeslab, f) )
  {
    cout << "Residual is too large, creating a new time slab." << endl;
    adaptivity.shift(u, f);
    u.reset();
    return false;
  }
  */

  // Update time
  t = timeslab.endtime();
  
  // Save solution
  save(timeslab);
  
  // Prepare for next time slab
  shift();
  
  // Update progress
  p = t / T;

  // Check if we are done
  if ( timeslab.finished() )
  {
    _finished = true;
    p = 1.0;
    cout << "Solution computed in " << toc() << " seconds." << endl;
    fixpoint.report();
  }

  return true;
}
//-----------------------------------------------------------------------------
void TimeStepper::shift()
{
  // Shift adaptivity
  adaptivity.shift(u, f);

  // Shift solution
  u.shift(t);
}
//-----------------------------------------------------------------------------
void TimeStepper::save(TimeSlab& timeslab)
{
  // Check if we should save the solution
  if ( !save_solution )
    return;

  // Choose method for saving the solution
  if ( adaptive_samples )
    saveAdaptiveSamples(timeslab);
  else
    saveFixedSamples(timeslab);
}
//-----------------------------------------------------------------------------
void TimeStepper::saveFixedSamples(TimeSlab& timeslab)
{
  // Compute time of first sample within time slab
  real K = T / static_cast<real>(no_samples);
  real t = ceil(timeslab.starttime()/K) * K;
  
  // Save samples
  while ( t < timeslab.endtime() )
  {
    Sample sample(u, f, t);
    file << sample;
    t += K;
  }
  
  // Save end time value
  if ( timeslab.finished() ) {
    Sample sample(u, f, timeslab.endtime());
    file << sample;
  }
}
//-----------------------------------------------------------------------------
void TimeStepper::saveAdaptiveSamples(TimeSlab& timeslab)
{
  // Get start time of time slab
  real t0 = timeslab.starttime();
  real t1 = t0;

  // Save initial value
  if ( t0 == 0.0 )
  {
    Sample sample(u, f, 0.0);
    file << sample;
  }

  // Create a list of element groups from the time slab
  ElementGroupList list(timeslab);

  // Save a samples within each element
  for (ElementGroupIterator group(list); !group.end(); ++group)
  {
    // Get first element in element group
    ElementIterator element(*group);
    
    // Check if we have stepped forward
    if ( element->endtime() <= t1 )
      continue;

    // Get interval
    t0 = element->starttime();
    t1 = element->endtime();

    dolfin_assert(sample_density >= 1);
    real dk = (t1 - t0) / static_cast<real>(sample_density);

    // Save samples
    for (unsigned int n = 0; n < sample_density; ++n)
    {
      // Compute time of sample, and make sure we get the end time right
      real t = t0 + n*dk;
      if ( n == (sample_density - 1) )
	t = t1;

      // Create and save the sample
      Sample sample(u, f, t0 + (static_cast<real>(n)+1.0)*dk);
      file << sample;
    }
  }
}
//-----------------------------------------------------------------------------
void TimeStepper::stabilize(real K)
{
  // Get stabilization parameters from fixed point iteration
  real alpha = 1.0;
  unsigned int m = 0;
  fixpoint.stabilization(alpha, m);

  // Compute stabilizing time step, at least (at most) a factor 1/2
  real k = std::min(alpha, 0.5) * K;

  // Stabilize
  adaptivity.stabilize(k, m);
}
//-----------------------------------------------------------------------------
