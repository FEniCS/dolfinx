// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __REGULATOR_H
#define __REGULATOR_H

#include <dolfin/constants.h>

namespace dolfin {

  /// Time step regulator.

  class Regulator {
  public:
    
    /// Constructor
    Regulator();

    /// Constructor
    Regulator(real k);

    /// Destructor
    ~Regulator();

    /// Initialize regulator (specify first time step)
    void init(real k);

    /// Update time step (set new desired value)
    void update(real k, real kmax, bool kfixed);

    /// Update maximum time step
    void update(real kmax);
    
    /// Return time step
    real timestep() const;

  private:

    real k;

  };

}

#endif
