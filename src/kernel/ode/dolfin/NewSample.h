// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_SAMPLE_H
#define __NEW_SAMPLE_H

#include <string>
#include <dolfin/constants.h>
#include <dolfin/Variable.h>

namespace dolfin
{

  class NewTimeSlab;

  /// Sample of solution values at a given point.

  class NewSample : public Variable
  {
  public:
    
    /// Constructor
    NewSample(NewTimeSlab& timeslab, real t, std::string name, std::string label);

    /// Destructor
    ~NewSample();

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

    NewTimeSlab& timeslab;
    real time;

  };

}

#endif
