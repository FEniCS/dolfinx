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

// Interface to ParMETIS parallel partitioner
namespace dolfinx::graph::ParMETIS
{
#ifdef HAS_PARMETIS
// Standard ParMETIS partition
AdjacencyList<std::int32_t>
partition(MPI_Comm mpi_comm, int n,
          const AdjacencyList<std::int64_t>& adj_graph, bool ghosting);

#endif
} // namespace dolfinx::graph::ParMETIS
