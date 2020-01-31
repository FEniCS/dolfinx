// Copyright (C) 2019 Igor A. Baratta
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Graph.h"
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <map>

namespace dolfinx
{

namespace graph
{

template <typename T>
class CSRGraph;

/// This class provides an interface to KaHIP parallel partitioner
class KaHIP
{
#ifdef HAS_KAHIP
public:
  // Standard KaHIP partition
  static std::pair<std::vector<int>, std::map<std::int64_t, std::vector<int>>>
  partition(MPI_Comm mpi_comm, int nparts,
            const CSRGraph<unsigned long long>& csr_graph);

#endif
};
} // namespace graph
} // namespace dolfinx
