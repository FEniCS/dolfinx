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

  /// The simple version of the time slab.

  class SimpleTimeSlab : public TimeSlab {
  public:
    
    /// Create time slab, including one iteration
    SimpleTimeSlab(real t0, real t1, Solution& u, Adaptivity& adaptivity);
    
    /// Destructor
    ~SimpleTimeSlab();
    
    /// Count the number of element groups contained in the time slab
    void countElementGroups(unsigned int& size);
    
    /// Add element groups contained in the time slab to the list
    void addElementGroups(NewArray<ElementGroup*>& groups, unsigned int& pos);

    /// Display structure of time slab
    void show(unsigned int depth = 0) const;

  private:
    
    // Create new time slab
    void create(Solution& u, Adaptivity& adaptivity);
    
  };

}

#endif
