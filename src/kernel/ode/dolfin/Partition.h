// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __PARTITION_H
#define __PARTITION_H

#include <functional>

#include <dolfin/constants.h>
#include <dolfin/NewArray.h>

namespace dolfin {

  class RHS;
  class TimeSteppingData;
  class Regulator;

  /// Partition is used in the recursive construction of time slabs
  /// and contains a list of component indices. The order of these
  /// indices determine which components should be in which time slab.

  class Partition {
  public:

    /// Constructor
    Partition(unsigned int N);

    /// Destructor
    ~Partition();

    /// Return size of partition
    int size() const;

    /// Return component index at given position
    int index(unsigned int pos) const;

    /// Update partition (reorder components starting at offset)
    void update(int offset, int& end, real& K, TimeSteppingData& data);
    
  private:

    // Debug partitioning
    void debug(unsigned int offset, unsigned int end, TimeSteppingData& data) const;

    // Compute largest time step
    real maximum(int offset, TimeSteppingData& data) const;

    // Comparison operator for the partition
    struct Less : public std::unary_function<unsigned int, bool> 
    {
      Less(real& K, TimeSteppingData& data);
      bool operator()(unsigned int index) const;
      
      real K;
      TimeSteppingData& data;
    };

    // List of component indices
    NewArray<unsigned int> indices;
    
    // Threshold for partition
    real threshold;
    
  };

}

#endif
