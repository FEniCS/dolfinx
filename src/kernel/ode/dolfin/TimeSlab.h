// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Updates by Johan Jansson 2003

#ifndef __TIME_SLAB_H
#define __TIME_SLAB_H

#include <vector>
#include <dolfin/dolfin_log.h>
#include <dolfin/Array.h>
#include <dolfin/Table.h>
#include <dolfin/constants.h>

namespace dolfin {

  class Element;
  class Partition;
  class Adaptivity;
  class RHS;
  class Solution;
  class FixedPointIteration;

  /// A TimeSlab represents (a subsystem of) the system of ODEs
  /// between synchronized time levels t0 and t1. 

  class TimeSlab {
  public:

    /// Create time slab, including one iteration
    TimeSlab(real t0, real t1);

    /// Destructor
    virtual ~TimeSlab();
    
    /// Update time slab (iteration)
    virtual real update(FixedPointIteration& fixpoint) = 0;

    /// Reset time slab to initial values
    virtual void reset(FixedPointIteration& fixpoint) = 0;

    /// Check if the given time is within the time slab
    bool within(real t) const;
    
    /// Check if the time slab reached the given end time
    bool finished() const;
    
    /// Return start time
    real starttime() const;
    
    /// Return end time
    real endtime() const;
    
   /// Return length of time slab
    real length() const;

    /// Compute maximum discrete residual in time slab
    virtual real computeMaxRd(Solution& u, RHS& f) = 0;

    /// Output
    friend LogStream& operator<<(LogStream& stream, const TimeSlab& timeslab);

  protected:
    
    // Specify and adjust the time step
    void setsize(real K, const Adaptivity& adaptivity);

    // Update elements (iteration)
    real updateElements(FixedPointIteration& fixpoint);

    /// Reset elements to initial values
    void resetElements(FixedPointIteration& fixpoint);

    /// Compute maximum discrete residual for elements
    real computeMaxRdElements(Solution& u, RHS& f);

    //--- Time slab data ---

    // Start and end time for time slab
    real t0;
    real t1;

    // True if we reached the given end time
    bool reached_endtime;

    // List of elements within this time slab
    std::vector<Element*> elements;

  };

}

#endif
