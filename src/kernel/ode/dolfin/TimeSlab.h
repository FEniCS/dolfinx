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
  class TimeSlabData;
  class TimeSteppingData;
  class Partition;
  class RHS;

  /// A TimeSlab represents (a subsystem of) the system of ODEs
  /// between synchronized time levels t0 and t1. 

  class TimeSlab {
  public:

    /// Create time slab, including one iteration
    TimeSlab(real t0, real t1);

    /// Destructor
    virtual ~TimeSlab();
    
    /// Update time slab (iteration)
    virtual void update(RHS& f, TimeSteppingData& newdata) = 0;

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

    /// Output
    friend LogStream& operator<<(LogStream& stream, const TimeSlab& timeslab);

  protected:
    
    // Specify and adjust the time step
    void setsize(real K, const TimeSteppingData& data);

    //--- Time slab data ---

    // Start and end time for time slab
    real t0;
    real t1;

    // True if we reached the given end time
    bool reached_endtime;

  };

}

#endif
