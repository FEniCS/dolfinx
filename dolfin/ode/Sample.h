// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-11-20
// Last changed: 2005

#ifndef __SAMPLE_H
#define __SAMPLE_H

#include <string>
#include <dolfin/main/constants.h>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  class TimeSlab;

  /// Sample of solution values at a given point.

  class Sample : public Variable
  {
  public:
    
    /// Constructor
    Sample(TimeSlab& timeslab, real t, std::string name, std::string label);

    /// Destructor
    ~Sample();

    /// Return number of components
    uint size() const;

    /// Return time t
    real t() const;

    /// Return value of component with given index
    real u(uint index);

    /// Return time step for component with given index
    real k(uint index);

    /// Return residual for component with given index
    real r(uint index);
    
  private:

    TimeSlab& timeslab;
    real time;

  };

}

#endif
