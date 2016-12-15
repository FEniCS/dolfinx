// Copyright (C) 2008-2009 Niclas Jansson, Ola Skavhaug, Anders Logg,
// Garth N. Wells and Chris Richardson
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

#ifndef __PARMETIS_PARTITIONER_H
#define __PARMETIS_PARTITIONER_H

#include <cstdint>
#include <cstddef>
#include <vector>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Set.h>
#include "CSRGraph.h"

namespace dolfin
{
  class CellType;

  /// This class provides an interface to ParMETIS

  class ParMETIS
  {
  public:

    /// Compute cell partition from local mesh data.  The output
    /// vector cell_partition contains the desired destination process
    /// numbers for each cell.  Cells shared on multiple processes
    /// have an entry in ghost_procs pointing to the set of sharing
    /// process numbers.  The mode argument determines which ParMETIS
    /// function is called. It can be one of "partition",
    /// "adaptive_repartition" or "refine". For meshes that have
    /// already been partitioned or are already well partitioned, it
    /// can be advantageous to use "adaptive_repartition" or "refine".
    static void compute_partition(const MPI_Comm mpi_comm,
            std::vector<int>& cell_partition,
            std::map<std::int64_t, std::vector<int>>& ghost_procs,
            const boost::multi_array<std::int64_t, 2>& cell_vertices,
            const std::size_t num_global_vertices,
            const CellType& cell_type,
            const std::string mode="partition");

  private:

#ifdef HAS_PARMETIS

    // Standard ParMETIS partition. CSRGraph should be const, but
    // ParMETIS accesses it non-const, so has to be non-const here
    template <typename T>
      static void partition(MPI_Comm mpi_comm,
                            CSRGraph<T>& csr_graph,
                            std::vector<int>& cell_partition,
                            std::map<std::int64_t, std::vector<int>>& ghost_procs);

    // ParMETIS adaptive repartition. CSRGraph should be const, but
    // ParMETIS accesses it non-const, so has to be non-const here
    template <typename T>
      static void adaptive_repartition(MPI_Comm mpi_comm,
                                       CSRGraph<T>& csr_graph,
                                       std::vector<int>& cell_partition);

    // ParMETIS refine repartition. CSRGraph should be const, but
    // ParMETIS accesses it non-const, so has to be non-const here
    template <typename T>
      static void refine(MPI_Comm mpi_comm, CSRGraph<T>& csr_graph,
                         std::vector<int>& cell_partition);
#endif


  };

}

#endif
