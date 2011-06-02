// Copyright (C) 2003-2009 Johan Jansson and Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Benjamin Kehlet 2008
//
// First added:  2003
// Last changed: 2009-09-08

#include <cmath>
#include <string>
#include <dolfin/io/File.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/constants.h>
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
  file(ode.parameters["solution_file_name"]),
  p("Time-stepping"),
  t(0),
  _stopped(false),
  save_solution(ode.parameters["save_solution"]),
  adaptive_samples(ode.parameters["adaptive_samples"]),
  num_samples(ode.parameters["number_of_samples"]),
  sample_density(ode.parameters["sample_density"]),
  save_to_odesolution(false),
  u(0)
{
  // Create time slab
  std::string method = ode.parameters["method"];
  if (method == "mcg" || method == "mdg")
  {
    timeslab = new MultiAdaptiveTimeSlab(ode);
  }
  else
  {
    timeslab = new MonoAdaptiveTimeSlab(ode);
  }
}
//-----------------------------------------------------------------------
TimeStepper::TimeStepper(ODE& ode, ODESolution& u) :
  ode(ode),
  timeslab(0),
  file(ode.parameters["solution_file_name"]),
  p("Time-stepping"),
  t(0),
  _stopped(false),
  save_solution(ode.parameters["save_solution"]),
  adaptive_samples(ode.parameters["adaptive_samples"]),
  num_samples(ode.parameters["number_of_samples"]),
  sample_density(ode.parameters["sample_density"]),
  save_to_odesolution(true),
  u(&u)
{
  // Create time slab
  std::string method = ode.parameters["method"];
  if (method == "mcg" || method == "mdg")
  {
    timeslab = new MultiAdaptiveTimeSlab(ode);
  }
  else
  {
    timeslab = new MonoAdaptiveTimeSlab(ode);
  }

  // initialize ODESolution object
  u.init(ode.size(),
	 timeslab->get_trial(),
	 timeslab->get_quadrature_weights());

}

//-----------------------------------------------------------------------
TimeStepper::~TimeStepper()
{
  delete timeslab;
}
//----------------------------------------------------------------------
void TimeStepper::solve()
{
  solve(0.0, ode.endtime());
}
//----------------------------------------------------------------------
void TimeStepper::solve(real t0, real t1)
{
  begin("Time-stepping over the time interval [%g, %g]",
        to_double(t0), to_double(t1));

  // Do time-stepping on [t0, t1]
  t = t0;
  while (!at_end(t, t1) && !_stopped)
  {
    // Make time step
    t = step();

    // Update progress
    p = to_double(t / ode.endtime());
  }

  end();
}
//-------------------------------------------------------------------------
real TimeStepper::step()
{
  return step(t, ode.endtime());
}
//-------------------------------------------------------------------------
real TimeStepper::step(real t0, real t1)
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
  save();

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
  assert(timeslab);
  timeslab->set_state(u);
}
//-----------------------------------------------------------------------------
void TimeStepper::get_state(real* u)
{
  assert(timeslab);
  timeslab->get_state(u);
}
//-----------------------------------------------------------------------------
void TimeStepper::save()
{
  //save to ODESolution object
  if (save_to_odesolution)
    timeslab->save_solution(*u);


  // Check if we should save the solution
  if (!save_solution)
    return;

  // Choose method for saving the solution
  if (adaptive_samples)
    save_adaptive_samples();
  else
    save_fixed_samples();
}
//-----------------------------------------------------------------------------
void TimeStepper::save_fixed_samples()
{
  // Get start time and end time of time slab
  real t0 = timeslab->starttime();
  real t1 = timeslab->endtime();

  // Save initial value
  if (t0 < real_epsilon())
    save_sample(0.0);

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

    save_sample(t);
  }

  // Save final value
  if (at_end(t1, ode.endtime()))
    save_sample(ode.endtime());
}
//-----------------------------------------------------------------------------
void TimeStepper::save_adaptive_samples()
{
  // Get start time and end time of time slab
  real t0 = timeslab->starttime();
  real t1 = timeslab->endtime();

  // Save initial value
  if (t0 < real_epsilon())
    save_sample(0.0);

  // Compute distance between samples
  assert(sample_density >= 1);
  real k = (t1 - t0) / static_cast<real>(sample_density);

  // Save samples
  for (unsigned int n = 0; n < sample_density; ++n)
  {
    real t = t0 + static_cast<real>(n + 1)*k;
    if (n == (sample_density - 1))
      t = t1;
    save_sample(t);
  }
}
//-----------------------------------------------------------------------------
void TimeStepper::save_sample(real t)
{
  // Create sample
  Sample sample(*timeslab, t, "u", "ODE solution");

  // Save to file
  file << sample;

  // Let user save sample (optional)
  ode.save(sample);
}
//-----------------------------------------------------------------------------
bool TimeStepper::at_end(real t, real T) const
{
  return T - t < real_epsilon();
}
//-----------------------------------------------------------------------------
