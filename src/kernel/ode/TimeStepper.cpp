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
#include <dolfin/Partition.h>
#include <dolfin/TimeSlabData.h>
#include <dolfin/SimpleTimeSlab.h>
#include <dolfin/RecursiveTimeSlab.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/TimeSlabSample.h>
#include <dolfin/TimeStepper.h>

using namespace dolfin;

/*
void storeSolution(TimeSlab &slab, TimeSlabData &data,
		   std::vector<std::pair<real, Vector> > &solution)
{
  Vector Ui(data.size());
  
  if(slab.timeslabs.size() == 0)
  {
    real t = slab.endtime();
    
    for (int i = 0; i < data.size(); i++)
    {
      Component &c = data.component(i);

      real value = c(t);
      //dolfin::cout << "U(" << t << ", " << i << "): " << value << dolfin::endl;

      Ui(i) = value;
    }

    solution.push_back(std::pair<real, Vector>(t, Ui));
    
  }
  else
  {
    for(std::vector<TimeSlab *>::iterator it = slab.timeslabs.begin();
	it != slab.timeslabs.end(); it++)
    {
      TimeSlab *s = *it;

      storeSolution(*s, data, solution);
    }
  }
}
*/

//-----------------------------------------------------------------------------
void TimeStepper::solve(ODE& ode, real t0, real t1)
{
  // Create time slab data
  TimeSlabData data(ode);

  // Create partition
  Partition partition(ode.size());

  // Create right-hand side
  RHS f(ode, data);

  // Create file for storing the solution
  File file("solution.m");

  // Get the number of output samples
  int no_samples = dolfin_get("number of samples");

  // Temporary way of storing solution
  std::vector<std::pair<real, Vector> > solution;
  Vector Ui(f.size());
  solution.push_back(std::pair<real, Vector>(t0, ode.u0));
  
  // Start time
  real t = t0;

  // The time slab
  TimeSlab* timeslab = 0;

  // The time-stepping loop
  Progress p("Time-stepping");
  while ( true ) {

    // Create a new time slab
    if ( t == t0 )
      timeslab = new SimpleTimeSlab(t, t1, f, data);
    else
      timeslab = new RecursiveTimeSlab(t, t1, f, data, partition, 0);
      
    cout << "Created a time slab: " << *timeslab << endl;

    // Iterate a couple of times on the time slab
    for (int i = 0; i < 2; i++)
      timeslab->update(f, data);

    // Save solution
    save(*timeslab, data, f, file, t0, t1, no_samples);
    
    // Check if we are done
    if ( timeslab->finished() )
      break;

    // Update time
    t = timeslab->endtime();

    // Update progress
    p = (t - t0) / (t1 - t0);

    // Shift solution at endtime to new u0
    data.shift(*timeslab);

    // Delete time slab
    delete timeslab;
  }

  // Temporary way of storing solution

  std::ofstream os;
  os.open("U.m", std::ios::out);

  typedef std::vector<std::pair<real, Vector> >::iterator dataiterator;

  os << "t" << " = [" << std::endl;

  for(dataiterator i = solution.begin(); i != solution.end(); i++)
  {
    real &arg = (*i).first;

    os << arg << "; ";
  }
  os << "]" << std::endl;


  os << "U" << " = [" << std::endl;
  for(dataiterator i = solution.begin(); i != solution.end(); i++)
  {
    Vector &res = (*i).second;

    os << "[";
    for ( unsigned int k = 0; k < res.size(); k++)
    {
      os << res(k) << ";";
    }
    os << "]";
    os << " ";
  }
  os << "]" << std::endl;

}
//-----------------------------------------------------------------------------
void TimeStepper::save(TimeSlab& timeslab, TimeSlabData& data, RHS& f,
		       File& file, real t0, real t1, int no_samples)
{
  // Compute time of first sample within time slab
  real K = (t1 - t0) / static_cast<real>(no_samples);
  real t =  t0 + ceil((timeslab.starttime()-t0)/K) * K;

  // Save samples
  while ( t < timeslab.endtime() )
  {
    // Create a sample of the solution
    TimeSlabSample sample(timeslab, data, f, t);
    
    // Save solution to file
    file << sample;

    // Step to next sample
    t += K;
  }
}
//-----------------------------------------------------------------------------
