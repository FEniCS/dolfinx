// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/ODE.h>
#include <dolfin/RHS.h>
#include <dolfin/Partition.h>
#include <dolfin/TimeSlabData.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/TimeStepper.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void TimeStepper::solve(ODE& ode, real t0, real t1)
{
  // Get size of the system
  int N = ode.size();

  // Create time slab data, partition and right-hand side
  TimeSlabData data(ode);
  Partition partition(N, 0.1);
  RHS f(ode, data);

  real t = t0;

  Progress p("Time-stepping");
  while ( true ) {

    dolfin_debug("stepping");


    // Create time slab
    TimeSlab timeslab(t, t1, f, data, partition, 0);

    // Iterate a couple of times on the time slab
    for (int i = 0; i < 3; i++)
      timeslab.update(f);

    // Check if we are done
    if ( timeslab.finished() )
      break;

    // Update time
    t = timeslab.endtime();

    // Update partition with new time steps
    //partition.update(data, 0);

    dolfin_debug("solution");
    for (int i = 0; i < data.size(); i++)
    {
      Component &c = data.component(i);

      real value = c(t);
      dolfin::cout << "U(" << t << ", " << i << ": " << value << dolfin::endl;
    }
  }

}
//-----------------------------------------------------------------------------
