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
///
/// An Algorithm for Reducing the Bandwidth and Profile of a Sparse Matrix
/// SIAM Journal on Numerical Analysis, Vol. 13, No. 2 (Apr., 1976), pp. 236-250
/// https://www.jstor.org/stable/2156090
///
/// @param graph Input graph to be analysed
/// @return Reordering vector
std::vector<int> gps_reorder(const graph::AdjacencyList<int>& graph);
} // namespace dolfinx::graph