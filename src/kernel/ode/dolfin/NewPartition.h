// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_PARTITION_H
#define __NEW_PARTITION_H

#include <functional>
#include <dolfin/constants.h>
#include <dolfin/NewArray.h>

namespace dolfin
{
  
  class MultiAdaptivity;

  /// Partition is used in the recursive construction of time slabs
  /// and contains a list of component indices. The order of these
  /// indices determine which components should be in which time slab.

  class NewPartition
  {
  public:

    /// Constructor
    NewPartition(uint N);

    /// Destructor
    ~NewPartition();

    /// Return size of partition
    uint size() const;

    /// Return component index at given position
    uint index(uint pos) const;

    /// Update partition (reorder components starting at offset)
    real update(uint offset, uint& end, MultiAdaptivity& adaptivity, real K);

    /// Debug partition
    void debug(uint offset, uint end) const;

  private:

    // Compute time step for partitioning
    real maxstep(uint offset, MultiAdaptivity& adaptivity) const;

    // Compute largest time step
    real maximum(uint offset, MultiAdaptivity& adaptivity) const;

    // Compute smallest time step
    real minimum(uint offset, uint end, MultiAdaptivity& adaptivity) const;

    // Update time steps
    void update(uint offset, uint end, MultiAdaptivity& adaptivity, real k) const;
    
    // Comparison operator for the partition
    struct Less : public std::unary_function<uint, bool> 
    {
      Less(real& K, MultiAdaptivity& adaptivity);
      bool operator()(uint index) const;
      
      real K;
      MultiAdaptivity& adaptivity;
    };

    // List of component indices
    NewArray<uint> indices;
    
    // Threshold for partition
    real threshold;
    
  };

}

#endif
