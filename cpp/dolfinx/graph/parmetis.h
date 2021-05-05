// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "partition.h"
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>

namespace dolfinx::graph::parmetis
{
#ifdef HAS_PARMETIS

/// Create a graph partitioning function that uses ParMETIS
///
/// param[in] options The ParMETIS option. See ParMETIS manual for
/// details.
graph::partition_fn partitioner(std::array<int, 3> options = {0, 0, 0});

#endif
} // namespace dolfinx::graph::parmetis
