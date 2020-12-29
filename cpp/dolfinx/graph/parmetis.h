// Copyright (C) 2008-2009 Niclas Jansson, Ola Skavhaug, Anders Logg,
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>

// Interface to ParMETIS parallel partitioner
namespace dolfinx::graph::parmetis
{
#ifdef HAS_PARMETIS
// Standard ParMETIS partition
AdjacencyList<std::int32_t> partition(MPI_Comm mpi_comm, int n,
                                      const AdjacencyList<std::int64_t>& graph,
                                      std::int32_t num_ghost_nodes,
                                      bool ghosting);

#endif
} // namespace dolfinx::graph::parmetis
