// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __PARTITION_H
#define __PARTITION_H

#include <vector>
#include <functional>

#include <dolfin/constants.h>
#include <dolfin/Array.h>

namespace dolfin {

  class RHS;
  class TimeSlabData;

  /// Partition is used in the recursive construction of time slabs
  /// and contains a list of component indices. The order of these
  /// indices determine which components should be in which time slab.

  class Partition {
  public:

    /// Constructor
    Partition(int N, real timestep);

    /// Destructor
    ~Partition();

    /// Return size of partition
    int size() const;

    /// Return component index at given position
    int index(unsigned int pos) const;

    /// Update data for partition (time steps)
    void update(int offset, RHS& f, TimeSlabData& data);

    /// Partition (reorder) components
    void partition(int offset, int& end, real& K);
    
    /// Invalidate partition (update needed)
    void invalidate();
    
  private:

    // Component index and time step
    class ComponentIndex {
    public:

      ComponentIndex() : index(0), timestep(0.0) {}

      int index;
      real timestep;

    };

    // Comparison operator for the partition
    struct Less : public std::unary_function<ComponentIndex, bool> 
    {
      Less(real& K) : K(K) {}

      bool operator()(ComponentIndex& component) const
      {
	return component.timestep >= K;
      }

      real K;
    };

    // List of component indices
    std::vector<ComponentIndex> components;
    
    // Compute largest time step
    real maximum(int offset) const;

    // State
    bool valid;

    // Threshold for partition
    real threshold;
    
  };

}

#endif
