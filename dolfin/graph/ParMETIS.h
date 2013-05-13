// Copyright (C) 2008-2009 Niclas Jansson, Ola Skavhaug and Anders Logg
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
// Modified by Garth N. Wells, 2010
//
// First added:  2010-02-10
// Last changed:

#ifndef __PARMETIS_PARTITIONER_H
#define __PARMETIS_PARTITIONER_H

#include <cstddef>
#include <vector>

namespace dolfin
{

  // Forward declarations
  class LocalMeshData;

  class ParMETISDualGraph;

  /// This class proivdes an interface to ParMETIS

  class ParMETIS
  {
  public:

    /// Compute cell partition using ParMETIS. For meshes that have
    /// already been partitioned or are already well partitioned,
    /// setting repartition=true attempts to balance the
    /// quality
    static void compute_partition(std::vector<std::size_t>& cell_partition,
                                  const LocalMeshData& mesh_data,
                                  bool repartition=false);

  private:

    // Standard ParMETIS partition
    static void partition(std::vector<std::size_t>& cell_partition,
                          ParMETISDualGraph& g);

    // ParMETIS adaptive repartition
    static void adaptive_repartition(std::vector<std::size_t>& cell_partition,
                                     ParMETISDualGraph& g);

    // ParMETIS refine repartition
    static void refine(std::vector<std::size_t>& cell_partition,
                       ParMETISDualGraph& g);

  };

}

#endif
