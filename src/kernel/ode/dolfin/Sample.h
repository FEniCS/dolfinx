// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SAMPLE_H
#define __SAMPLE_H

#include <dolfin/constants.h>

namespace dolfin {

  class TimeSteppingData;
  class RHS;

  /// Sample of values at a given point.

  class Sample {
  public:

    /// Constructor
    Sample(TimeSteppingData& data, RHS& f, real t);

    /// Destructor
    ~Sample();

    /// Return number of components
    unsigned int size() const;

    /// Return time t
    real t() const;

    /// Return value of component with given index
    real u(unsigned int index);

    /// Return time step for component with given index
    real k(unsigned int index);

    /// Return residual for component with given index
    real r(unsigned int index);
    
  private:

    TimeSteppingData& data;
    RHS& f;
    real time;

  };

}

#endif
