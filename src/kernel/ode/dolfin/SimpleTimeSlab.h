// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SIMPLE_TIME_SLAB_H
#define __SIMPLE_TIME_SLAB_H

#include <dolfin/Element.h>
#include <dolfin/TimeSlab.h>

namespace dolfin {

  class Adaptivity;
  class RHS;
  class Solution;
  class FixedPointIteration;

  /// The simple version of the time slab.

  class SimpleTimeSlab : public TimeSlab {
  public:
    
    /// Create time slab, including one iteration
    SimpleTimeSlab(real t0, real t1, Solution& u, Adaptivity& adaptivity);
    
    /// Destructor
    ~SimpleTimeSlab();
    
    /// Update time slab (iteration)
    void update(FixedPointIteration& fixpoint);
    
    /// Reset time slab to initial values
    void reset(FixedPointIteration& fixpoint);

    /// Check if the time slab is a leaf
    bool leaf() const;

    /// Compute L2 norm of element residual
    real elementResidualL2(FixedPointIteration& fixpoint);

    /// Display structure of time slab
    void show(unsigned int depth = 0) const;

  private:
    
    // Create new time slab
    void create(Solution& u, Adaptivity& adaptivity);
    
  };

}

#endif
