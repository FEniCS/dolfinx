// Copyright (C) 2008-2009 Niclas Jansson, Ola Skavhaug, Anders Logg,
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>
#include <dolfin/common/MPI.h>
#include <dolfin/common/types.h>
#include <dolfin/mesh/PartitionData.h>
#include <string>
#include <vector>

namespace dolfin
{

namespace mesh
{
class CellType;
}

namespace graph
{

template <typename T>
class CSRGraph;

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
  static mesh::PartitionData
  compute_partition(const MPI_Comm mpi_comm,
                    const Eigen::Ref<const EigenRowArrayXXi64> cell_vertices,
                    const mesh::CellType& cell_type,
                    const std::string mode = "partition");

private:
#ifdef HAS_PARMETIS

  // Standard ParMETIS partition. CSRGraph should be const, but
  // ParMETIS accesses it non-const, so has to be non-const here
  template <typename T>
  static mesh::PartitionData partition(MPI_Comm mpi_comm,
                                       const CSRGraph<T>& csr_graph);

  // ParMETIS adaptive repartitiont, so has to be non-const here
  template <typename T>
  static std::vector<int> adaptive_repartition(MPI_Comm mpi_comm,
                                               const CSRGraph<T>& csr_graph,
                                               double weight = 1000);

  // ParMETIS refine repartition
  template <typename T>
  static std::vector<int> refine(MPI_Comm mpi_comm,
                                 const CSRGraph<T>& csr_graph);
#endif
};
} // namespace graph
} // namespace dolfin
