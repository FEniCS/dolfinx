// Copyright (C) 2010 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <vector>

#define BOOST_NO_HASH

#include <boost/graph/adjacency_list.hpp>
#include <dolfin/common/Set.h>

namespace dolfin
{

/// Typedefs for simple graph data structures

/// DOLFIN container for graphs
typedef dolfin::Set<int> graph_set_type;
/// Vector of unordered Sets
typedef std::vector<graph_set_type> Graph;
}


