// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SIMPLE_TIME_SLAB_H
#define __SIMPLE_TIME_SLAB_H

#include <dolfin/TimeSlab.h>

namespace dolfin {

  class Element;
  class Adaptivity;
  class RHS;
  class Solution;

  /// The simple version of the time slab.

  class SimpleTimeSlab : public TimeSlab {
  public:
    
    /// Create time slab, including one iteration
    SimpleTimeSlab(real t0, real t1, Solution& u, Adaptivity& adaptivity);
    
    /// Destructor
    ~SimpleTimeSlab();
    
    /// Update time slab (iteration)
    void update(Solution& u, RHS& f);

  private:
    
    // Create new time slab
    void create(Solution& u, Adaptivity& adaptivity);
    
  };

}

#endif
