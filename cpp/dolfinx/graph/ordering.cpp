// Copyright (C) 2021-2026 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "ordering.h"
#include "AdjacencyList.h"
#include <algorithm>
#include <cstdint>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <limits>
#include <span>

using namespace dolfinx;

namespace
{
//-----------------------------------------------------------------------------
// Create a level structure from graph, rooted at node s
graph::AdjacencyList<int>
create_level_structure(const graph::AdjacencyList<int>& graph, int s)
{
  common::Timer t("Graph: create_level_structure");

  // Note: int8 is often faster than bool. A fresh buffer is
  // allocated on each call (rather than a reused scratch buffer)
  // since the allocator's bulk zero-fill is faster than resetting
  // only the touched entries when a large fraction of the graph is
  // typically visited, as is the case for compact/dense mesh graphs.
  std::vector<std::int8_t> labelled(graph.num_nodes(), false);
  labelled[s] = true;

  // Current level
  int l = 0;

  std::vector<int> level_offsets{0};
  level_offsets.reserve(graph.offsets().size());
  std::vector<int> level_structure = {s};
  level_structure.reserve(graph.array().size());
  while (static_cast<int>(level_structure.size()) > level_offsets.back())
  {
    level_offsets.push_back(level_structure.size());
    for (int i = level_offsets[l]; i < level_offsets[l + 1]; ++i)
    {
      const int node = level_structure[i];
      for (int idx : graph.links(node))
      {
        if (labelled[idx])
          continue;
        level_structure.push_back(idx);
        labelled[idx] = true;
      }
    }
    ++l;
  }

  return graph::AdjacencyList(std::move(level_structure),
                              std::move(level_offsets));
}
//-----------------------------------------------------------------------------
// Reverse Cuthill-McKee algorithm, finding a reordering for the given
// graph, operating only on nodes which are yet unlabelled (indicated
// with -1 in the vector rlabel).
std::vector<std::int32_t>
rcm_reorder_unlabelled(const graph::AdjacencyList<std::int32_t>& graph,
                       std::span<const std::int32_t> rlabel)
{
  common::Timer timer("Reverse Cuthill-McKee ordering");

  const std::int32_t n = graph.num_nodes();

  // Degree comparison function
  auto cmp_degree = [&graph](auto a, auto b)
  { return graph.num_links(a) < graph.num_links(b); };

  // Pick an arbitrary vertex of minimal degree and call it v
  std::int32_t v = 0;
  std::int32_t dmin = std::numeric_limits<std::int32_t>::max();
  for (std::int32_t i = 0; i < n; ++i)
  {
    if (int d = graph.num_links(i); rlabel[i] == -1 and d < dmin)
    {
      v = i;
      dmin = d;
    }
  }

  // Find a pseudo-peripheral root: repeatedly move to the minimum-degree
  // vertex of the deepest level found so far (the classical George-Liu
  // "double sweep"), stopping once no deeper level structure is found.
  graph::AdjacencyList<int> lv = create_level_structure(graph, v);
  bool done = false;
  while (!done)
  {
    auto lv_final = lv.links(lv.num_nodes() - 1);
    int s = *std::ranges::min_element(lv_final, cmp_degree);
    graph::AdjacencyList<int> lstmp = create_level_structure(graph, s);
    if (lstmp.num_nodes() > lv.num_nodes())
    {
      v = s;
      lv = std::move(lstmp);
    }
    else
      done = true;
  }

  // Cuthill-McKee numbering, breadth-first from the root: each time a
  // vertex is processed, its not-yet-discovered neighbours are appended
  // in increasing degree order. Note this sorts each vertex's own
  // newly-discovered neighbours as they are found, not each BFS level
  // as a whole group -- the latter is a common but lower-quality
  // simplification, since it discards the parent/discovery order that
  // the standard algorithm relies on.
  std::vector<std::int8_t> labelled(n, false);
  std::vector<int> rv;
  rv.reserve(n);
  rv.push_back(v);
  labelled[v] = true;

  std::vector<int> nbr;
  for (std::size_t current = 0; current < rv.size(); ++current)
  {
    nbr.clear();
    for (int w : graph.links(rv[current]))
    {
      if (!labelled[w])
      {
        nbr.push_back(w);
        labelled[w] = true;
      }
    }
    std::ranges::sort(nbr, cmp_degree);
    rv.insert(rv.end(), nbr.begin(), nbr.end());
  }

  // Reverse the numbering -- the "reverse" in Reverse Cuthill-McKee --
  // which tends to reduce profile relative to plain Cuthill-McKee.
  std::ranges::reverse(rv);

  return rv;
}

} // namespace

//-----------------------------------------------------------------------------
std::vector<std::int32_t>
graph::reorder_rcm(const graph::AdjacencyList<std::int32_t>& graph)
{
  const std::int32_t n = graph.num_nodes();
  std::vector<std::int32_t> r(n, -1);
  std::vector<std::int32_t> rv;

  // Repeat for each disconnected part of the graph
  int count = 0;
  while (count < n)
  {
    rv = rcm_reorder_unlabelled(graph, r);
    assert(!rv.empty());

    // Reverse permutation
    for (std::int32_t q : rv)
      r[q] = count++;
  }

  // Check all labelled
  assert(std::find(r.begin(), r.end(), -1) == r.end());
  return r;
}
//-----------------------------------------------------------------------------
