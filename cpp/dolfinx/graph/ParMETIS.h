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

namespace dolfinx::graph
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

#endif
};
} // namespace dolfinx::graph
