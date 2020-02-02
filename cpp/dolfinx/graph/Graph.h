// Copyright (C) 2010 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/Set.h>
#include <vector>

namespace dolfinx
{
namespace graph
{

/// Typedefs for simple graph data structures

/// DOLFINX container for graphs
typedef dolfinx::common::Set<int> graph_set_type;

/// Vector of unordered Sets
typedef std::vector<graph_set_type> Graph;

} // namespace graph
} // namespace dolfinx
