// Copyright (C) 2008-2009 Niclas Jansson, Ola Skavhaug, Anders Logg,
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Graph.h"
#include <cstddef>
#include <cstdint>
#include <dolfin/common/MPI.h>
#include <dolfin/common/types.h>
#include <map>
#include <string>
#include <utility>
#include <vector>

// FIXME: Avoid exposing ParMETIS publicly
#ifdef HAS_PARMETIS
#include <parmetis.h>
#endif

namespace dolfin
{

namespace graph
{

template <typename T>
class CSRGraph;

/// This class provides an interface to ParMETIS

class ParMETIS
{
#ifdef HAS_PARMETIS
public:
  // Standard ParMETIS partition
  static std::pair<std::vector<int>, std::map<std::int64_t, std::vector<int>>>
  partition(MPI_Comm mpi_comm, idx_t nparts, const CSRGraph<idx_t>& csr_graph);

private:
  // ParMETIS adaptive repartition, so has to be non-const here
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
