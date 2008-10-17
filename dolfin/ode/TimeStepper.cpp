// Copyright (C) 2003-2008 Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Benjamin Kehlet 2008
//
// First added:  2003
// Last changed: 2008-10-06

#include <cmath>
#include <string>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/timing.h>
#include <dolfin/common/constants.h>
#include <dolfin/parameter/parameters.h>
#include "ODE.h"
#include "Sample.h"
#include "MonoAdaptiveTimeSlab.h"
#include "MultiAdaptiveTimeSlab.h"
#include "TimeStepper.h"

using namespace dolfin;

//--------------------------------------------------------------------------
TimeStepper::TimeStepper(ODE& ode, ODESolution& u) :
  N(ode.size()), t(0), T(ode.endtime()),
  ode(ode), u(u),
  timeslab(0), file(ode.get("ODE solution file name")),
  p("Time-stepping"), _stopped(false), _finished(false),
  save_solution(ode.get("ODE save solution")),
  adaptive_samples(ode.get("ODE adaptive samples")),
  no_samples(ode.get("ODE number of samples")),
  sample_density(ode.get("ODE sample density"))
{
  // Create time slab
  std::string method = ode.get("ODE method");
  if ( method == "mcg" || method == "mdg" )
  {
    timeslab = new MultiAdaptiveTimeSlab(ode);
  }
  else    
  {
    timeslab = new MonoAdaptiveTimeSlab(ode);
  }
}
//-----------------------------------------------------------------------
TimeStepper::~TimeStepper()
{
  delete timeslab;
}
//----------------------------------------------------------------------
void TimeStepper::solve(ODE& ode, ODESolution& u)
{
  // Start timing
  tic();  
  
  // Create a time stepper object
  TimeStepper timeStepper(ode, u);
    
  // Do time stepping
  while (!timeStepper.finished() && !timeStepper.stopped())
    timeStepper.step();
  
  // Report elapsed time
  message("Solution computed in %.3f seconds.", toc());
}
//-------------------------------------------------------------------------
real TimeStepper::step()
{
  // FIXME: Change type of time slab if solution does not converge

  // Check if this is the first time step
  const bool first = t < ODE::epsilon();

  // Reset stopped flag
  _stopped = false;

  // Iterate until solution is accepted
  const real a = t;
  while (true)
  {
    // Build time slab
    t = timeslab->build(a, T);
    //timeslab->disp();
    
    // Solve time slab system
    if ( !timeslab->solve() )
    {
      _stopped = true;
      break;
    }
    //timeslab->disp();

    // Check if solution can be accepted
    if ( timeslab->check(first) )
      break;
    
    message("Rejecting time slab K = %.3e, trying again.", to_double( timeslab->length() ));
  }

  // Save solution
  save();

  // Check if solution was stopped
  if ( _stopped )
    warning("Solution stopped at t = %.3e.", to_double(t));

  // Update for next time slab
  if ( !timeslab->shift(finished()) )
  {
    message("ODE solver stopped on user's request.");
    _stopped = true;
  }

  // Update progress
  if (!_stopped)
    p = to_double(t / T);

  return t;
}
//-----------------------------------------------------------------------------
bool TimeStepper::finished() const
{
  return t >= T;
}
//-----------------------------------------------------------------------------
bool TimeStepper::stopped() const
{
  return _stopped;
}
//-----------------------------------------------------------------------------
void TimeStepper::save()
{
  // Check if we should save the solution
  if (!save_solution)
    return;

  // Choose method for saving the solution
  if (adaptive_samples)
    saveAdaptiveSamples();
  else
    saveFixedSamples();
}
//-----------------------------------------------------------------------------
void TimeStepper::saveFixedSamples()
{
  // Get start time and end time of time slab
  real t0 = timeslab->starttime();
  real t1 = timeslab->endtime();

  // Save initial value
  if (t0 == 0.0)
  {
    //Sample sample(*timeslab, 0.0, u.name(), u.label());
    Sample sample(*timeslab, ode.time(0.0), "u", "unknown");
    file << sample;
    u.add_sample(sample);
    ode.save(sample);
  }

  // Compute distance between samples
  real K = T / static_cast<real>(no_samples);
  real t = floor(t0/K - 0.5) * K;

  // Save samples
  while (true)
  {
    t += K;

    if ( (t - ODE::epsilon()) < t0 )
      continue;

    if ( (t - ODE::epsilon()) > t1 )
      break;

    if ( abs(t - t1) < ODE::epsilon() )
      t = t1;

    Sample sample(*timeslab, ode.time(t), "u", "unknown");
    file << sample;
    u.add_sample(sample);
    ode.save(sample);
  }

  // Save final value
  if (t1 == T)
  {
    //Sample sample(*timeslab, 0.0, u.name(), u.label());
    Sample sample(*timeslab, ode.time(T), "u", "unknown");
    file << sample;
    u.add_sample(sample);
    ode.save(sample);
  }
}
//-----------------------------------------------------------------------------
void TimeStepper::saveAdaptiveSamples()
{
  // Get start time and end time of time slab
  real t0 = timeslab->starttime();
  real t1 = timeslab->endtime();

  // Save initial value
  if ( t0 == 0.0 )
  {
    //Sample sample(*timeslab, 0.0, u.name(), u.label());
    Sample sample(*timeslab, ode.time(0.0), "u", "unknown");
    file << sample;
    u.add_sample(sample);
    ode.save(sample);
  }

  // Compute distance between samples
  dolfin_assert(sample_density >= 1);
  real k = (t1 - t0) / static_cast<real>(sample_density);
  
  // Save samples
  for (unsigned int n = 0; n < sample_density; ++n)
  {
    // Compute time of sample, and make sure we get the end time right
    real t = t0 + static_cast<real>(n + 1)*k;
    if ( n == (sample_density - 1) )
      t = t1;
    
    // Create and save the sample
    //Sample sample(*timeslab, t, u.name(), u.label());
    Sample sample(*timeslab, ode.time(t), "u", "unknown");
    file << sample;
    u.add_sample(sample);
    ode.save(sample);
  }
}
//-----------------------------------------------------------------------------
