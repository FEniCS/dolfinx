// Copyright (C) 2021 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/graph/AdjacencyList.h>
#include <vector>

namespace dolfinx::graph
{
/// Implementation of the Gibbs-Poole-Stockmeyer algorithm
/// @param graph Input graph to be analysed
/// @return Reordering vector
std::vector<int> gps_reorder(const graph::AdjacencyList<int>& graph);
} // namespace dolfinx::graph