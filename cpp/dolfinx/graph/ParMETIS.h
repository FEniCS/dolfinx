// Copyright (C) 2008-2009 Niclas Jansson, Ola Skavhaug, Anders Logg,
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <vector>

// FIXME: Avoid exposing ParMETIS publicly
#ifdef HAS_PARMETIS
#include <parmetis.h>
#endif

namespace dolfinx
{

namespace graph
{

/// This class provides an interface to ParMETIS

class ParMETIS
{
#ifdef HAS_PARMETIS
public:
  // Standard ParMETIS partition
  static AdjacencyList<std::int32_t>
  partition(MPI_Comm mpi_comm, idx_t nparts,
            const AdjacencyList<idx_t>& adj_graph, bool ghosting);

private:
  // ParMETIS adaptive repartition, so has to be non-const here
  template <typename T>
  static std::vector<int>
  adaptive_repartition(MPI_Comm mpi_comm, const AdjacencyList<T>& adj_graph,
                       double weight = 1000);

  // ParMETIS refine repartition
  template <typename T>
  static std::vector<int> refine(MPI_Comm mpi_comm,
                                 const AdjacencyList<T>& adj_graph);
#endif
};
} // namespace graph
} // namespace dolfinx
