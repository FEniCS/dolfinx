// Copyright (C) 2004-2005 Johan Jansson and Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2004
// Last changed: 2005

#ifndef __PARTITION_H
#define __PARTITION_H

#include <functional>
#include <dolfin/common/types.h>
#include <dolfin/common/real.h>
#include <dolfin/common/real.h>

namespace dolfin
{

  class MultiAdaptivity;

  /// Partition is used in the recursive construction of time slabs
  /// and contains a list of component indices. The order of these
  /// indices determine which components should be in which time slab.

  class Partition
  {
  public:

    /// Constructor
    Partition(uint N);

    /// Destructor
    ~Partition();

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
    std::vector<uint> indices;

    // Threshold for partition
    real threshold;

  };

}

#endif
