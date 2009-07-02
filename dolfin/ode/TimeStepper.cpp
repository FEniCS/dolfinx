// Copyright (C) 2003-2009 Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Benjamin Kehlet 2008
//
// First added:  2003
// Last changed: 2009-02-10

#include <cmath>
#include <string>
#include <dolfin/io/File.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/constants.h>
#include <dolfin/parameter/parameters.h>
#include "ODE.h"
#include "Sample.h"
#include "MonoAdaptiveTimeSlab.h"
#include "MultiAdaptiveTimeSlab.h"
#include "TimeStepper.h"

using namespace dolfin;

//--------------------------------------------------------------------------
TimeStepper::TimeStepper(ODE& ode) :
  ode(ode),
  timeslab(0),
  file(ode.parameters("solution_file_name")),
  p("Time-stepping"),
  t(0),
  _stopped(false),
  save_solution(ode.parameters("save_solution")),
  adaptive_samples(ode.parameters("adaptive_samples")),
  num_samples(ode.parameters("number_of_samples")),
  sample_density(ode.parameters("sample_density"))
{
  // Create time slab
  std::string method = ode.parameters("method");
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
void TimeStepper::solve(ODESolution& u)
{
  solve(u, 0.0, ode.endtime());
}
//----------------------------------------------------------------------
void TimeStepper::solve(ODESolution& u, real t0, real t1)
{
  begin("Time-stepping over the time interval [%g, %g]",
        to_double(t0), to_double(t1));

  // Do time-stepping on [t0, t1]
  t = t0;
  while (!at_end(t, t1) && !_stopped)
  {
    // Make time step
    t = step(u);

    // Update progress
    p = to_double(t / ode.endtime());
  }

  end();
}
//-------------------------------------------------------------------------
real TimeStepper::step(ODESolution& u)
{
  return step(u, t, ode.endtime());
}
//-------------------------------------------------------------------------
real TimeStepper::step(ODESolution& u, real t0, real t1)
{
  // FIXME: Change type of time slab if solution does not converge

  // Check if this is the first time step
  const bool first = t < real_epsilon();

  // Iterate until solution is accepted
  _stopped = false;
  while (true)
  {
    // Build time slab
    t = timeslab->build(t0, t1);

    // Try to solve time slab system
    if (!timeslab->solve())
    {
      warning("ODE solver did not converge at t = %g", to_double(t));
      _stopped = true;
      break;
    }

    // Check if solution can be accepted
    if (timeslab->check(first))
      break;

    info("Rejecting time slab K = %.3e, trying again.", to_double(timeslab->length()));
  }

  // Save solution
  save(u);

  // Update for next time slab
  if (!timeslab->shift(at_end(t, ode.endtime())))
  {
    info("ODE solver stopped on user's request at t = %g.", to_double(t));
    _stopped = true;
  }

  return t;
}
//-----------------------------------------------------------------------------
void TimeStepper::set_state(const real* u)
{
  dolfin_assert(timeslab);
  timeslab->set_state(u);
}
//-----------------------------------------------------------------------------
void TimeStepper::get_state(real* u)
{
  dolfin_assert(timeslab);
  timeslab->get_state(u);
}
//-----------------------------------------------------------------------------
void TimeStepper::save(ODESolution& u)
{
  // Check if we should save the solution
  if (!save_solution)
    return;

  // Choose method for saving the solution
  if (adaptive_samples)
    save_adaptive_samples(u);
  else
    save_fixed_samples(u);
}
//-----------------------------------------------------------------------------
void TimeStepper::save_fixed_samples(ODESolution& u)
{
  // Get start time and end time of time slab
  real t0 = timeslab->starttime();
  real t1 = timeslab->endtime();

  // Save initial value
  if (t0 < real_epsilon())
    save_sample(u, 0.0);

  // Compute distance between samples
  real K = ode.endtime() / static_cast<real>(num_samples);
  real t = floor(t0 / K - 0.5) * K;

  // Save samples
  while (true)
  {
    t += K;

    if (t < t0 + real_epsilon())
      continue;

    if (t > t1 + real_epsilon())
      break;

    if (real_abs(t - t1) < real_epsilon())
      t = t1;

    save_sample(u, t);
  }

  // Save final value
  if (at_end(t1, ode.endtime()))
    save_sample(u, ode.endtime());
}
//-----------------------------------------------------------------------------
void TimeStepper::save_adaptive_samples(ODESolution& u)
{
  // Get start time and end time of time slab
  real t0 = timeslab->starttime();
  real t1 = timeslab->endtime();

  // Save initial value
  if (t0 < real_epsilon())
    save_sample(u, 0.0);

  // Compute distance between samples
  dolfin_assert(sample_density >= 1);
  real k = (t1 - t0) / static_cast<real>(sample_density);

  // Save samples
  for (unsigned int n = 0; n < sample_density; ++n)
  {
    real t = t0 + static_cast<real>(n + 1)*k;
    if (n == (sample_density - 1))
      t = t1;
    save_sample(u, t);
  }
}
//-----------------------------------------------------------------------------
void TimeStepper::save_sample(ODESolution& u, real t)
{
  // Create sample
  Sample sample(*timeslab, t, "u", "ODE solution");

  // Save to file
  file << sample;

  // Add sample to ODE solution
  u.add_sample(sample);

  // Let user save sample (optional)
  ode.save(sample);
}
//-----------------------------------------------------------------------------
bool TimeStepper::at_end(real t, real T) const
{
  return T - t < real_epsilon();
}
//-----------------------------------------------------------------------------
