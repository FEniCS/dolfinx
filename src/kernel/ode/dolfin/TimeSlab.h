// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Updates by Johan Jansson 2003

#ifndef __TIME_SLAB_H
#define __TIME_SLAB_H

#include <vector>

#include <dolfin/Array.h>
#include <dolfin/Table.h>
#include <dolfin/constants.h>

namespace dolfin {

  class Element;
  class TimeSlabData;
  class Partition;
  class RHS;

  /// A TimeSlab represents (a subsystem of) the system of ODEs
  /// between synchronized time levels t0 and t1. 

  class TimeSlab {
  public:

    /// Create time slab, including one iteration
    TimeSlab(real t0, real t1, RHS& f, 
	     TimeSlabData& data, Partition& partition, int offset);

    /// Destructor
    ~TimeSlab();

    /// Update time slab (iteration)
    void update(RHS& f, TimeSlabData& data);

    /// Update time slab (iteration)
    void updateu0(TimeSlab &prevslab);

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
    
  private:
    
    // Create new time slab
    void create(RHS& f, TimeSlabData& data, Partition& partition, int offset);
    
    // Create list of time slabs within the time slab
    void createTimeSlabs(RHS& f, TimeSlabData& data,
			 Partition& partition, int offset);

    // Create list of elements within the time slab
    void createElements(RHS& f, TimeSlabData& data,
			Partition& partition, int offset, int end);

    // Update time slabs (iteration)
    void updateTimeSlabs(RHS& f, TimeSlabData& data);

    // Update elements (iteration)
    void updateElements(RHS& f, TimeSlabData& data);

    // Specify and adjust the time step
    void setsize(real K);

    // Add a new time slab
    void add(TimeSlab* timeslab);

    // Start and end time for time slab
    real t0;
    real t1;

    // True if we reached the given end time
    bool reached_endtime;

    // List of elements within this time slab
    //Table<Element>::Iterator first;
    //Table<Element>::Iterator last;

    std::vector<Element*> elements;
    
    // List of time slabs within this time slab
    std::vector<TimeSlab*> timeslabs;

  };

}

#endif
