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
#include <dolfin/SimpleTimeSlab.h>
#include <dolfin/RecursiveTimeSlab.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/TimeSteppingData.h>
#include <dolfin/Sample.h>
#include <dolfin/Solution.h>
#include <dolfin/TimeStepper.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void TimeStepper::solve(ODE& ode)
{
  unsigned int no_samples = dolfin_get("number of samples");
  unsigned int N = ode.size();
  real T = ode.endtime();
  real t = 0.0;

  // Create data for time-stepping
  Partition partition(N);
  TimeSteppingData data(N);
  ElementData elmdata(N);
  Solution solution(ode, elmdata);
  RHS f(ode, solution);
  TimeSlab* timeslab = 0;

  // Create file for storing the solution
  File file("solution.m");

  // The time-stepping loop
  Progress p("Time-stepping");
  while ( true ) {
    
    // Create a new time slab
    if ( t == 0.0 )
      timeslab = new SimpleTimeSlab(t, T, f, data, solution);
    else
      timeslab = new RecursiveTimeSlab(t, T, f, data, partition, 0, solution);
    
    //cout << "Created a time slab: " << *timeslab << endl;
    
    // Iterate a couple of times on the time slab
    for (int i = 0; i < 2; i++)
      timeslab->update(f, data, solution);

    // Save solution
    save(solution, f, *timeslab, file, T, no_samples);

    // Update time
    t = timeslab->endtime();

    // Update progress
    p = t / T;

    // Prepare for next time slab
    shift(solution, f, data, t);

    // Check if we are done
    if ( timeslab->finished() )
    {
      delete timeslab;
      break;
    }

    // Delete time slab
    delete timeslab;
  }

}
//-----------------------------------------------------------------------------
void TimeStepper::shift(Solution& solution, RHS& f, TimeSteppingData& data, 
			real t)
{
  real TOL = data.tolerance();
  real kmax = data.maxstep();

  // Update residuals and time steps
  for (unsigned int i = 0; i < solution.size(); i++)
  {
    // Get last element
    Element* element = solution.last(i);
    dolfin_assert(element);

    // Compute residual
    real r = element->computeResidual(f);

    // Compute new time step
    real k = element->computeTimeStep(TOL, r, kmax);

    // Update regulator
    data.regulator(i).update(k, kmax);
  }
  
  // Shift solution
  solution.shift(t);
}
//-----------------------------------------------------------------------------
void TimeStepper::save(Solution& solution, RHS& f, TimeSlab& timeslab,
		       File& file, real T, unsigned int no_samples)
{
  // Compute time of first sample within time slab
  real K = T / static_cast<real>(no_samples);
  real t = ceil(timeslab.starttime()/K) * K;

  // Save samples
  while ( t < timeslab.endtime() )
  {
    Sample sample(solution, f, t);    
    file << sample;
    t += K;
  }
  
  // Save end time value
  if ( timeslab.finished() ) {
    Sample sample(solution, f, timeslab.endtime());
    file << sample;
  }
}
//-----------------------------------------------------------------------------
