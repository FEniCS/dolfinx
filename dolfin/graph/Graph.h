// Copyright (C) 2010 Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2010-11-16
// Last changed: 2010-11-25

#ifndef __GRAPH_TYPES_H
#define __GRAPH_TYPES_H

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

#endif
