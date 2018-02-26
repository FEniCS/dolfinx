// Copyright (C) 2008-2009 Niclas Jansson, Ola Skavhaug, Anders Logg,
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <boost/multi_array.hpp>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "CSRGraph.h"
#include <dolfin/common/MPI.h>

namespace dolfin
{
class CellType;

namespace graph
{

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
  static void
  compute_partition(const MPI_Comm mpi_comm, std::vector<int>& cell_partition,
                    std::map<std::int64_t, std::vector<int>>& ghost_procs,
                    const boost::multi_array<std::int64_t, 2>& cell_vertices,
                    const std::size_t num_global_vertices,
                    const CellType& cell_type,
                    const std::string mode = "partition");

private:
#ifdef HAS_PARMETIS

  // Standard ParMETIS partition. CSRGraph should be const, but
  // ParMETIS accesses it non-const, so has to be non-const here
  template <typename T>
  static void partition(MPI_Comm mpi_comm, CSRGraph<T>& csr_graph,
                        std::vector<int>& cell_partition,
                        std::map<std::int64_t, std::vector<int>>& ghost_procs);

  // ParMETIS adaptive repartition. CSRGraph should be const, but
  // ParMETIS accesses it non-const, so has to be non-const here
  template <typename T>
  static void adaptive_repartition(MPI_Comm mpi_comm, CSRGraph<T>& csr_graph,
                                   std::vector<int>& cell_partition);

  // ParMETIS refine repartition. CSRGraph should be const, but
  // ParMETIS accesses it non-const, so has to be non-const here
  template <typename T>
  static void refine(MPI_Comm mpi_comm, CSRGraph<T>& csr_graph,
                     std::vector<int>& cell_partition);
#endif
};
}
}