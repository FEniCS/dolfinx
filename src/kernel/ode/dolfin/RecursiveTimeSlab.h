// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Updates by Johan Jansson 2003

#ifndef __RECURSIVE_TIME_SLAB_H
#define __RECURSIVE_TIME_SLAB_H

#include <dolfin/TimeSlab.h>

namespace dolfin {

  class Element;
  class Adaptivity;
  class Partition;
  class RHS;
  class Solution;

  /// The recursive version of the time slab.

  class RecursiveTimeSlab : public TimeSlab {
  public:

    /// Create time slab, including one iteration
    RecursiveTimeSlab(real t0, real t1, Solution& u, RHS& f, 
		      Adaptivity& adaptivity, Partition& partition, int offset);
    
    /// Destructor
    ~RecursiveTimeSlab();
    
    /// Update time slab (iteration)
    void update(Solution& u, RHS& f);

  private:
    
    // Create new time slab
    void create(Solution& u, RHS& f, Adaptivity& adaptivity,
		Partition& partition, int offset);
    
    // Create list of time slabs within the time slab
    void createTimeSlabs(Solution& u, RHS& f, Adaptivity& adaptivity,
			 Partition& partition, int offset);

    // Create list of elements within the time slab
    void createElements(Solution& u, RHS& f, Adaptivity& adaptivity,
			Partition& partition, int offset, int end);

    // Update time slabs (iteration)
    void updateTimeSlabs(Solution& u, RHS& f);
    
    // Compute residuals and new time steps
    void computeResiduals(RHS& f, Adaptivity& adaptivity);

    //--- Time slab data ---

    // List of time slabs within this time slab
    std::vector<TimeSlab*> timeslabs;

  };

}

#endif
