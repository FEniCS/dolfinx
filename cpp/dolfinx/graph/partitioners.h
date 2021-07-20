// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "partition.h"
#include <cstdint>
#include <dolfinx/graph/AdjacencyList.h>
#include <functional>
#include <mpi.h>

namespace dolfinx::graph
{

namespace parmetis
{
#ifdef HAS_PARMETIS

/// Create a graph partitioning function that uses ParMETIS
///
/// param[in] options The ParMETIS option. See ParMETIS manual for
/// details.
graph::partition_fn partitioner(std::array<int, 3> options = {0, 0, 0});

#endif
} // namespace parmetis

/// Interfaces to KaHIP parallel partitioner
namespace kahip
{
#ifdef HAS_KAHIP
/// Create a graph partitioning function that uses KaHIP
///
/// @param[in] mode The KaHiP partitioning mode (see
/// https://github.com/KaHIP/KaHIP/blob/master/parallel/parallel_src/interface/parhip_interface.h)
/// @param[in] seed The KaHiP random number generator seed
/// @param[in] imbalance The allowable imbalance
/// @param[in] suppress_output Suppresses KaHIP output if true
/// @return A KaHIP graph partitioning function with specified parameter
/// options
graph::partition_fn partitioner(int mode = 1, int seed = 0,
                                double imbalance = 0.03,
                                bool suppress_output = true);

#endif
} // namespace kahip
} // namespace dolfinx::graph