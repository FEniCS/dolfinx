// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// - Updates by Johan Jansson (2003)

#include <algorithm>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cmath>

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/File.h>
#include <dolfin/ODE.h>
#include <dolfin/RHS.h>
#include <dolfin/ElementData.h>
#include <dolfin/Partition.h>
#include <dolfin/Adaptivity.h>
#include <dolfin/Solution.h>
#include <dolfin/SimpleTimeSlab.h>
#include <dolfin/RecursiveTimeSlab.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/Sample.h>
#include <dolfin/TimeStepper.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void TimeStepper::solve(ODE& ode, Function& function)
{
  unsigned int no_samples = dolfin_get("number of samples");
  unsigned int N = ode.size();
  real T = ode.endtime();
  real t = 0.0;
  TimeSlab* timeslab = 0;

  // Create data for time-stepping
  Partition partition(N);
  Adaptivity adaptivity(N);
  Solution u(ode, function);
  RHS f(ode, u);

  // Create file for storing the solution
  File file(u.label() + ".m");

  // The time-stepping loop
  Progress p("Time-stepping");
  while ( true ) {
    
    // Create a new time slab
    if ( t == 0.0 )
      timeslab = new SimpleTimeSlab(t, T, u, adaptivity);
    else
      timeslab = new RecursiveTimeSlab(t, T, u, f, adaptivity, partition, 0);
    
    // Iterate a couple of times on the time slab
    for (int i = 0; i < 2; i++)
      timeslab->update(u, f);

    // Update time
    t = timeslab->endtime();
    
    // Save solution
    save(u, f, *timeslab, file, T, no_samples);

    // Prepare for next time slab
    shift(u, f, adaptivity, t);

    // Check if we are done
    if ( timeslab->finished() )
    {
      delete timeslab;
      break;
    }

    // Delete time slab
    delete timeslab;

    // Update progress
    p = t / T;
  }

}
//-----------------------------------------------------------------------------
void TimeStepper::shift(Solution& u, RHS& f, Adaptivity& adaptivity, real t)
{
  real TOL = adaptivity.tolerance();
  real kmax = adaptivity.maxstep();

  // Update residuals and time steps
  for (unsigned int i = 0; i < u.size(); i++)
  {
    // Get last element
    Element* element = u.last(i);
    dolfin_assert(element);

    // Compute residual
    real r = element->computeResidual(f);

    // Compute new time step
    real k = element->computeTimeStep(TOL, r, kmax);

    // Update regulator
    adaptivity.regulator(i).update(k, kmax);
  }
  
  // Shift solution
  u.shift(t);
}
//-----------------------------------------------------------------------------
void TimeStepper::save(Solution& u, RHS& f, TimeSlab& timeslab,
		       File& file, real T, unsigned int no_samples)
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
