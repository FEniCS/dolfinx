// Copyright (C) 2019 Igor A. Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <dolfinx/graph/AdjacencyList.h>
#include <functional>
#include <mpi.h>

/// Interfaces to KaHIP parallel partitioner
namespace dolfinx::graph::kahip
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
std::function<graph::AdjacencyList<std::int32_t>(
    MPI_Comm, int, const AdjacencyList<std::int64_t>&, std::int32_t, bool)>
partitioner(int mode = 1, int seed = 0, double imbalance = 0.03,
            bool suppress_output = true);

#endif
} // namespace dolfinx::graph::kahip
