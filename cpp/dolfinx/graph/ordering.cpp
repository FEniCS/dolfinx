// Copyright (C) 2021 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "ordering.h"
#include <algorithm>
#include <cstdint>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <limits>
#include <span>

using namespace dolfinx;

namespace
{
//-----------------------------------------------------------------------------
// Compute the sets of connected components of the input "graph" which
// contain the nodes in "indices".
std::vector<std::vector<int>>
residual_graph_components(const graph::AdjacencyList<int>& graph,
                          std::span<const int> indices)
{
  if (indices.empty())
    return std::vector<std::vector<int>>();

  const int n = graph.num_nodes();

  // Mark all nodes as labelled, except those in the residual graph
  std::vector<std::int_fast8_t> labelled(n, true);
  for (int w : indices)
    labelled[w] = false;

  // Find first unlabelled entry
  auto it = std::find(labelled.begin(), labelled.end(), false);

  std::vector<std::vector<int>> rgc;
  std::vector<int> r;
  r.reserve(n);
  while (it != labelled.end())
  {
    r.clear();
    r.push_back(std::distance(labelled.begin(), it));
    labelled[r.front()] = true;

    // Get connected component of graph starting from r[0]
    std::size_t c = 0;
    while (c < r.size())
    {
      for (int w : graph.links(r[c]))
      {
        if (!labelled[w])
        {
          r.push_back(w);
          labelled[w] = true;
        }
      }
      ++c;
    }
    rgc.push_back(r);

    // Find next unlabelled entry
    it = std::find(it, labelled.end(), false);
  }

  std::sort(rgc.begin(), rgc.end(),
            [](const std::vector<int>& a, const std::vector<int>& b)
            { return (a.size() > b.size()); });

  return rgc;
}
//-----------------------------------------------------------------------------
// Get the (maximum) width of a level structure
int max_level_width(const graph::AdjacencyList<int>& levels)
{
  int wmax = 0;
  for (int i = 0; i < levels.num_nodes(); ++i)
    wmax = std::max(wmax, levels.num_links(i));
  return wmax;
}
//-----------------------------------------------------------------------------
// Create a level structure from graph, rooted at node s
graph::AdjacencyList<int>
create_level_structure(const graph::AdjacencyList<int>& graph, int s)
{
  common::Timer t("GPS: create_level_structure");

  // Note: int8 is often faster than bool
  std::vector<std::int8_t> labelled(graph.num_nodes(), false);
  labelled[s] = true;

  // Current level
  int l = 0;

  std::vector<int> level_offsets = {0};
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

  return graph::AdjacencyList<int>(std::move(level_structure),
                                   std::move(level_offsets));
}

//-----------------------------------------------------------------------------
// Gibbs-Poole-Stockmeyer algorithm, finding a reordering for the given
// graph, operating only on nodes which are yet unlabelled (indicated
// with -1 in the vector rlabel).
std::vector<std::int32_t>
gps_reorder_unlabelled(const graph::AdjacencyList<std::int32_t>& graph,
                       std::span<const std::int32_t> rlabel)
{
  common::Timer timer("Gibbs-Poole-Stockmeyer ordering");

  const int n = graph.num_nodes();

  // Degree comparison function
  auto cmp_degree = [&graph](int a, int b)
  { return (graph.num_links(a) < graph.num_links(b)); };

  // ALGORITHM I. Finding endpoints of a pseudo-diameter.

  // A. Pick an arbitrary vertex of minimal degree and call it v
  int v = 0;
  int dmin = std::numeric_limits<int>::max();
  for (int i = 0; i < n; ++i)
  {
    if (int d = graph.num_links(i); d < dmin and rlabel[i] == -1)
    {
      v = i;
      dmin = d;
    }
  }

  // B. Generate a level structure Lv rooted at vertex v.
  graph::AdjacencyList<int> lv = create_level_structure(graph, v);
  graph::AdjacencyList<int> lu(0);
  bool done = false;
  int u = 0;
  while (!done)
  {
    // Sort final level S of Lv into increasing degree order
    auto lv_final = lv.links(lv.num_nodes() - 1);
    std::vector<int> S(lv_final.size());
    std::partial_sort_copy(lv_final.begin(), lv_final.end(), S.begin(), S.end(),
                           cmp_degree);

    int w_min = std::numeric_limits<int>::max();
    done = true;

    // C. Generate level structures rooted at vertices s in S selected
    // in order of increasing degree.
    for (int s : S)
    {
      auto lstmp = create_level_structure(graph, s);
      if (lstmp.num_nodes() > lv.num_nodes())
      {
        // Found a deeper level structure, so restart
        v = s;
        lv = lstmp;
        done = false;
        break;
      }

      //  D. Let u be the vertex of S whose associated level structure
      //  has smallest width
      if (int w = max_level_width(lstmp); w < w_min)
      {
        w_min = w;
        u = s;
        lu = lstmp;
      }
    }
  }

  // If degree of u is less than v, swap
  if (graph.num_links(u) < graph.num_links(v))
  {
    std::swap(u, v);
    std::swap(lu, lv);
  }

  assert(lv.num_nodes() == lu.num_nodes());
  int k = lv.num_nodes();
  LOG(INFO) << "GPS pseudo-diameter:(" << k << ") " << u << "-" << v << "\n";

  // ALGORITHM II. Minimizing level width.

  // Level pair (i, j) associated with each node
  std::vector<std::array<int, 2>> lvp(n);
  for (int i = 0; i < k; ++i)
  {
    for (int w : lv.links(i))
      lvp[w][0] = i;
    for (int w : lu.links(i))
      lvp[w][1] = k - 1 - i;
  }

  assert(lvp[v][0] == 0 and lvp[v][1] == 0);
  assert(lvp[u][0] == (k - 1) and lvp[u][1] == (k - 1));

  // Insert any nodes (i, i) into new level structure ls and capture
  // residual nodes in rg
  std::vector<std::vector<int>> ls(k);
  std::vector<int> rg;
  for (int i = 0; i < k; ++i)
  {
    for (int w : lu.links(i))
    {
      if (lvp[w][0] == lvp[w][1])
        ls[lvp[w][0]].push_back(w);
      else
        rg.push_back(w);
    }
  }

  {
    const std::vector<std::vector<int>> rgc
        = residual_graph_components(graph, rg);

    // Width of levels with additional entries from rgc
    std::vector<int> wn(k), wh(k), wl(k);

    for (const std::vector<int>& r : rgc)
    {
      std::transform(ls.begin(), ls.end(), wn.begin(),
                     [](const std::vector<int>& vec) { return vec.size(); });
      std::copy(wn.begin(), wn.end(), wh.begin());
      std::copy(wn.begin(), wn.end(), wl.begin());
      for (int w : r)
      {
        ++wh[lvp[w][0]];
        ++wl[lvp[w][1]];
      }
      // Zero any entries which did not increase
      std::transform(wh.begin(), wh.end(), wn.begin(), wh.begin(),
                     [](int vh, int vn) { return (vh > vn) ? vh : 0; });
      std::transform(wl.begin(), wl.end(), wn.begin(), wl.begin(),
                     [](int vl, int vn) { return (vl > vn) ? vl : 0; });

      // Find maximum of those that did increase
      int h0 = *std::max_element(wh.begin(), wh.end());
      int l0 = *std::max_element(wl.begin(), wl.end());

      // Choose which side to use
      int side = h0 < l0 ? 0 : 1;

      // If h0 == l0, then use the elements of the level pairs which arise
      // from the rooted level structure of smaller width. If the widths are
      // equal, use the first elements. (i.e. lvp[][0]).
      if (h0 == l0)
        side = max_level_width(lu) < max_level_width(lv) ? 1 : 0;

      for (int w : r)
        ls[lvp[w][side]].push_back(w);
    }
  }

  // ALGORITHM III. Numbering
  std::vector<int> rv;
  rv.reserve(n);
  std::vector<std::int8_t> labelled(n, false);

  int current_node = 0;
  rv.push_back(v);
  labelled[v] = true;

  // Temporary work vectors
  std::vector<std::int8_t> in_level;
  std::vector<int> rv_next;
  std::vector<int> nbr, nbr_next;
  std::vector<int> nrem;

  for (const std::vector<int>& lslevel : ls)
  {
    // Mark all nodes of the current level
    in_level.assign(n, false);
    for (int w : lslevel)
      in_level[w] = true;

    rv_next.clear();
    while (true)
    {
      while (current_node < static_cast<int>(rv.size()))
      {
        // Get unlabelled neighbours of current node in this level and
        // next level
        nbr.clear();
        nbr_next.clear();
        for (int w : graph.links(rv[current_node]))
        {
          if (labelled[w])
            continue;

          if (in_level[w])
            nbr.push_back(w);
          else
            nbr_next.push_back(w);
        }

        // Add nodes to rv in order of increasing degree
        std::sort(nbr.begin(), nbr.end(), cmp_degree);
        rv.insert(rv.end(), nbr.begin(), nbr.end());
        for (int w : nbr)
          labelled[w] = true;

        // Save nodes for next level to a separate list, rv_next
        std::sort(nbr_next.begin(), nbr_next.end(), cmp_degree);
        rv_next.insert(rv_next.end(), nbr_next.begin(), nbr_next.end());
        for (int w : nbr_next)
          labelled[w] = true;

        ++current_node;
      }

      // Find any remaining unlabelled nodes in level and label the one
      // with lowest degree
      nrem.clear();
      for (int w : lslevel)
        if (!labelled[w])
          nrem.push_back(w);

      if (nrem.size() == 0)
        break;

      std::sort(nrem.begin(), nrem.end(), cmp_degree);
      rv.push_back(nrem.front());
      labelled[nrem.front()] = true;
    }

    // Insert already-labelled nodes of next level
    rv.insert(rv.end(), rv_next.begin(), rv_next.end());
  }

  return rv;
}

} // namespace

//-----------------------------------------------------------------------------
std::vector<std::int32_t>
graph::reorder_gps(const graph::AdjacencyList<std::int32_t>& graph)
{
  const std::int32_t n = graph.num_nodes();
  std::vector<std::int32_t> r(n, -1);
  std::vector<std::int32_t> rv;

  // Repeat for each disconnected part of the graph
  int count = 0;
  while (count < n)
  {
    rv = gps_reorder_unlabelled(graph, r);
    assert(rv.size() > 0);

    // Reverse permutation
    for (std::int32_t q : rv)
      r[q] = count++;
  }

  // Check all labelled
  assert(std::find(r.begin(), r.end(), -1) == r.end());
  return r;
}
//-----------------------------------------------------------------------------
