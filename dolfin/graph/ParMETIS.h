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

#include <cstdint>
#include <cstddef>
#include <vector>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Set.h>

namespace dolfin
{

  // Forward declarations
  class LocalMeshData;

  class ParMETISDualGraph;

  /// This class provides an interface to ParMETIS

  class ParMETIS
  {
  public:

    /// Compute cell partition from local mesh data.
    /// The output vector cell_partition contains the desired
    /// destination process numbers for each cell.
    /// Cells shared on multiple processes have an
    /// entry in ghost_procs pointing to
    /// the set of sharing process numbers.
    /// The mode argument determines which ParMETIS function
    /// is called. It can be one of "partition",
    /// "adaptive_repartition" or "refine". For meshes
    /// that have already been partitioned or are already well
    /// partitioned, it can be advantageous to use
    /// "adaptive_repartition" or "refine".
    static void compute_partition(const MPI_Comm mpi_comm,
            std::vector<std::size_t>& cell_partition,
            std::map<std::int64_t, dolfin::Set<int>>& ghost_procs,
            const LocalMeshData& mesh_data,
            std::string mode="partition");

  private:

#ifdef HAS_PARMETIS
    // Standard ParMETIS partition
    static void partition(MPI_Comm mpi_comm,
       std::vector<std::size_t>& cell_partition,
       std::map<std::int64_t, dolfin::Set<int>>& ghost_procs,
       ParMETISDualGraph& g);

    // ParMETIS adaptive repartition
    static void adaptive_repartition(MPI_Comm mpi_comm,
                                     std::vector<std::size_t>& cell_partition,
                                     ParMETISDualGraph& g);

    // ParMETIS refine repartition
    static void refine(MPI_Comm mpi_comm,
                       std::vector<std::size_t>& cell_partition,
                       ParMETISDualGraph& g);
#endif


  };

}

#endif
