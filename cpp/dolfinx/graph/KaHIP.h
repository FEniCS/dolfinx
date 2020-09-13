// Copyright (C) 2019 Igor A. Baratta
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <dolfinx/graph/AdjacencyList.h>
#include <mpi.h>

// Interface to KaHIP parallel partitioner
namespace dolfinx::graph::KaHIP
{
#ifdef HAS_KAHIP
// Standard KaHIP partition
AdjacencyList<std::int32_t>
partition(MPI_Comm mpi_comm, int nparts,
          const AdjacencyList<unsigned long long>& adj_graph, bool ghosting);

#endif
} // namespace dolfinx::graph::KaHIP
