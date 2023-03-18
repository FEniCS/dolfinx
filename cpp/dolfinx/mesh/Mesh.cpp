// Copyright (C) 2006-2020 Anders Logg, Chris Richardson, Jorgen S.
// Dokken and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Mesh.h"
#include "Geometry.h"
#include "Topology.h"
#include "cell_types.h"
#include "graphbuild.h"
#include "topologycomputation.h"
#include "utils.h"
#include <algorithm>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/ordering.h>
#include <dolfinx/graph/partition.h>
#include <memory>

using namespace dolfinx;
using namespace dolfinx::mesh;

namespace
{
// /// Re-order an adjacency list
// template <typename T>
// graph::AdjacencyList<T> reorder_list(const graph::AdjacencyList<T>& list,
//                                      std::span<const std::int32_t> nodemap)
// {
//   // Copy existing data to keep ghost values (not reordered)
//   std::vector<T> data(list.array());
//   std::vector<std::int32_t> offsets(list.offsets().size());

//   // Compute new offsets (owned and ghost)
//   offsets[0] = 0;
//   for (std::size_t n = 0; n < nodemap.size(); ++n)
//     offsets[nodemap[n] + 1] = list.num_links(n);
//   for (std::size_t n = nodemap.size(); n < (std::size_t)list.num_nodes(); ++n)
//     offsets[n + 1] = list.num_links(n);
//   std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
//   graph::AdjacencyList<T> list_new(std::move(data), std::move(offsets));

//   for (std::size_t n = 0; n < nodemap.size(); ++n)
//   {
//     auto links_old = list.links(n);
//     auto links_new = list_new.links(nodemap[n]);
//     assert(links_old.size() == links_new.size());
//     std::copy(links_old.begin(), links_old.end(), links_new.begin());
//   }

//   return list_new;
// }
} // namespace

//-----------------------------------------------------------------------------
