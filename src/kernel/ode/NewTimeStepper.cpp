// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Jansson 2003, 2004.

#include <cmath>
#include <string>
#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/timeinfo.h>
#include <dolfin/ODE.h>
#include <dolfin/ReducedModel.h>
#include <dolfin/NewSample.h>
#include <dolfin/MonoAdaptiveTimeSlab.h>
#include <dolfin/MultiAdaptiveTimeSlab.h>
#include <dolfin/NewTimeStepper.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewTimeStepper::NewTimeStepper(ODE& ode) :
  N(ode.size()), t(0), T(ode.endtime()),
  ode(ode), timeslab(0), file(dolfin_get("file name")),
  p("Time-stepping"), stopped(false), _finished(false),
  save_solution(dolfin_get("save solution")),
  solve_dual(dolfin_get("solve dual problem")),
  adaptive_samples(dolfin_get("adaptive samples")),
  no_samples(dolfin_get("number of samples")),
  sample_density(dolfin_get("sample density"))
{
  dolfin_warning("ODE solver is EXPERIMENTAL.");

  // Create time slab
  std::string method = dolfin_get("method");
  if ( method == "mcg" || method == "mdg" )
  {
    timeslab = new MultiAdaptiveTimeSlab(ode);
  }
  else    
    timeslab = new MonoAdaptiveTimeSlab(ode);
}
//-----------------------------------------------------------------------------
NewTimeStepper::~NewTimeStepper()
{
  if ( timeslab ) delete timeslab;
}
//-----------------------------------------------------------------------------
void NewTimeStepper::solve(ODE& ode)
{
  // Start timing
  tic();  
  
  // Check if we should create a reduced model (automatic modeling)
  if ( dolfin_get("automatic modeling") )
  {
    dolfin_error("Not implemented.");

    dolfin_info("Creating reduced model (automatic modeling).");

    // Create the reduced model
    //ReducedModel reducedModel(ode);

    // Create a time stepper object
    //NewTimeStepper timeStepper(reducedModel, u);
    
    // Do time stepping
    //while ( !timeStepper.finished() )
    //  timeStepper.step();
  }
  else
  {
    // Create a time stepper object
    NewTimeStepper timeStepper(ode);
    
    // Do time stepping
    while ( !timeStepper.finished() && !timeStepper.stopped )
      timeStepper.step();
  }

  // Report elapsed time
  cout << "Solution computed in " << toc() << " seconds." << endl;
}
//-----------------------------------------------------------------------------
real NewTimeStepper::step()
{
  bool solved = false;

  while(!solved)
  {
    // Build time slab
    t = timeslab->build(t, T);
    
    //timeslab->disp();
    
    // Solve time slab system
    solved = timeslab->solve();
  }

  // Save solution
  save();

  // Update for next time slab
  if ( !timeslab->shift() )
  {
    dolfin_info("ODE solver stopped on user's request.");
    stopped = true;
  }

  // Update progress
  p = t / T;

  return t;
}
//-----------------------------------------------------------------------------
bool NewTimeStepper::finished() const
{
  return t >= T;
}
//-----------------------------------------------------------------------------
void NewTimeStepper::save()
{
  // Check if we should save the solution
  if ( !save_solution )
    return;

  // Choose method for saving the solution
  if ( adaptive_samples )
    saveAdaptiveSamples();
  else
    saveFixedSamples();
}
//-----------------------------------------------------------------------------
void NewTimeStepper::saveFixedSamples()
{
  // Get start time and end time of time slab
  real t0 = timeslab->starttime();
  real t1 = timeslab->endtime();

  // Save initial value
  if ( t0 == 0.0 )
  {
    //NewSample sample(*timeslab, 0.0, u.name(), u.label());
    NewSample sample(*timeslab, 0.0, "u", "unknown");
    file << sample;
    ode.save(sample);
  }

  // Compute distance between samples
  real K = T / static_cast<real>(no_samples);
  real t = floor(t0/K - 0.5) * K;

  // Save samples
  while ( true )
  {
    t += K;

    if ( (t - DOLFIN_EPS) < t0 )
      continue;

    if ( (t - DOLFIN_EPS) > t1 )
      break;

    if ( fabs(t - t1) < DOLFIN_EPS )
      t = t1;

    //NewSample sample(*timeslab, t, u.name(), u.label());
    NewSample sample(*timeslab, t, "u", "uknown");
    file << sample;
    ode.save(sample);
  }
}
//-----------------------------------------------------------------------------
void NewTimeStepper::saveAdaptiveSamples()
{
  // Get start time and end time of time slab
  real t0 = timeslab->starttime();
  real t1 = timeslab->endtime();

  // Save initial value
  if ( t0 == 0.0 )
  {
    //NewSample sample(*timeslab, 0.0, u.name(), u.label());
    NewSample sample(*timeslab, 0.0, "u", "unknown");
    file << sample;
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
    //NewSample sample(*timeslab, t, u.name(), u.label());
    NewSample sample(*timeslab, t, "u", "unknown");
    file << sample;
    ode.save(sample);
  }
}
//-----------------------------------------------------------------------------
