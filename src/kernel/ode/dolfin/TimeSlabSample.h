// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TIME_SLAB_SAMPLE_H
#define __TIME_SLAB_SAMPLE_H

#include <dolfin/constants.h>

namespace dolfin {

  class TimeSlab;
  class TimeSteppingData;
  class RHS;

  /// A TimeSlabSample is sample of the values at a given point
  /// within a given time slab.

  class TimeSlabSample {
  public:

    /// Constructor
    TimeSlabSample(TimeSlab& timeslab, TimeSteppingData& data, 
		   RHS& f, real t);

    /// Destructor
    ~TimeSlabSample();

    /// Return number of components
    unsigned int size() const;

    /// Return time t
    real time() const;

    /// Return value of component with given index
    real value(unsigned int index);

    /// Return time step for component with given index
    real timestep(unsigned int index);

    /// Return residual for component with given index
    real residual(unsigned int index);
    
  private:

    TimeSlab& timeslab;
    TimeSteppingData& data;
    RHS& f;
    real t;

  };

}

#endif
