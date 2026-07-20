// Copyright (C) 2021-2026 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "ordering.h"
#include "AdjacencyList.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <limits>
#include <span>
#include <thread>

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

  std::ranges::sort(rgc,
                    [](const std::vector<int>& a, const std::vector<int>& b)
                    { return (a.size() > b.size()); });

  return rgc;
}
//-----------------------------------------------------------------------------
// Get the (maximum) width of a level structure
std::size_t max_level_width(const graph::AdjacencyList<int>& levels)
{
  const std::vector<std::int32_t>& offsets = levels.offsets();
  return std::transform_reduce(
      offsets.begin(), std::prev(offsets.end()), std::next(offsets.begin()),
      std::size_t(0),
      [](auto x0, auto x1) -> std::size_t { return std::max(x1, x0); },
      [](auto x0, auto x1) -> std::size_t { return x1 - x0; });
}
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
// Gibbs-Poole-Stockmeyer algorithm, finding a reordering for the given
// graph, operating only on nodes which are yet unlabelled (indicated
// with -1 in the vector rlabel).
std::vector<std::int32_t>
gps_reorder_unlabelled(const graph::AdjacencyList<std::int32_t>& graph,
                       std::span<const std::int32_t> rlabel,
                       std::size_t max_candidates, std::size_t num_threads)
{
  common::Timer timer("Gibbs-Poole-Stockmeyer ordering");

  const std::int32_t n = graph.num_nodes();

  // Degree comparison function
  auto cmp_degree = [&graph](auto a, auto b)
  { return graph.num_links(a) < graph.num_links(b); };

  // ALGORITHM I. Finding endpoints of a pseudo-diameter.

  // A. Pick an arbitrary vertex of minimal degree and call it v
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

  // B. Generate a level structure Lv rooted at vertex v.
  graph::AdjacencyList<int> lv = create_level_structure(graph, v);
  graph::AdjacencyList<int> lu(0);
  bool done = false;
  int u = 0;
  std::vector<int> S;
  while (!done)
  {
    // Sort final level S of Lv into increasing degree order, capped to
    // at most `max_candidates` (most likely peripheral) candidates.
    // Testing every vertex in the final level costs O(|S| * (V+E)) per
    // round; for a compact 3D mesh |S| scales as the cross-sectional
    // area (~ N^(2/3)), which dominates run time at scale. A handful
    // of the lowest-degree candidates often captures the same
    // pseudo-diameter and width-minimising choice as an exhaustive
    // search, but not always -- non-convex domains and strongly graded
    // meshes can lose some bandwidth/profile quality with a small cap.
    auto lv_final = lv.links(lv.num_nodes() - 1);
    S.resize(std::min(lv_final.size(), max_candidates));
    std::partial_sort_copy(lv_final.begin(), lv_final.end(), S.begin(), S.end(),
                           cmp_degree);
    int w_min = std::numeric_limits<int>::max();
    done = true;

    // C. Generate level structures rooted at vertices s in S selected
    // in order of increasing degree. Each candidate's level structure
    // is independent of the others, so when num_threads > 1 they are
    // computed concurrently. The decision of which candidate to use
    // is still made afterwards, sequentially, in the original order,
    // so the result is identical regardless of num_threads.
    std::vector<graph::AdjacencyList<int>> lstmps(S.size(),
                                                  graph::AdjacencyList<int>(0));
    std::size_t nt = std::min(num_threads, S.size());
    if (nt <= 1)
    {
      for (std::size_t i = 0; i < S.size(); ++i)
        lstmps[i] = create_level_structure(graph, S[i]);
    }
    else
    {
      std::vector<std::jthread> workers;
      workers.reserve(nt);
      std::size_t chunk = (S.size() + nt - 1) / nt;
      for (std::size_t t = 0; t < nt; ++t)
      {
        std::size_t begin = t * chunk;
        std::size_t end = std::min(S.size(), begin + chunk);
        if (begin >= end)
          break;
        workers.emplace_back(
            [&S, &lstmps, &graph, begin, end]()
            {
              for (std::size_t i = begin; i < end; ++i)
                lstmps[i] = create_level_structure(graph, S[i]);
            });
      }
    }

    for (std::size_t i = 0; i < S.size(); ++i)
    {
      int s = S[i];
      graph::AdjacencyList<int>& lstmp = lstmps[i];
      if (lstmp.num_nodes() > lv.num_nodes())
      {
        // Found a deeper level structure, so restart
        v = s;
        lv = std::move(lstmp);
        done = false;
        break;
      }

      //  D. Let u be the vertex of S whose associated level structure
      //  has smallest width
      if (int w = max_level_width(lstmp); w < w_min)
      {
        w_min = w;
        u = s;
        lu = std::move(lstmp);
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
  const int k = lv.num_nodes();
  spdlog::info("GPS pseudo-diameter:({}) {}-{}", k, u, v);

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
      if (auto lvp0 = lvp[w][0]; lvp0 == lvp[w][1])
        ls[lvp0].push_back(w);
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
      std::ranges::transform(ls, wn.begin(), [](const std::vector<int>& vec)
                             { return vec.size(); });
      std::ranges::copy(wn, wh.begin());
      std::ranges::copy(wn, wl.begin());
      for (int w : r)
      {
        ++wh[lvp[w][0]];
        ++wl[lvp[w][1]];
      }
      // Zero any entries which did not increase
      std::ranges::transform(wh, wn, wh.begin(),
                             [](int vh, int vn) { return (vh > vn) ? vh : 0; });
      std::ranges::transform(wl, wn, wl.begin(),
                             [](int vl, int vn) { return (vl > vn) ? vl : 0; });

      // Find maximum of those that did increase
      int h0 = std::ranges::max(wh);
      int l0 = std::ranges::max(wl);

      // Choose which side to use
      int side = h0 < l0 ? 0 : 1;

      // If h0 == l0, then use the elements of the level pairs which
      // arise from the rooted level structure of smaller width. If the
      // widths are equal, use the first elements. (i.e. lvp[][0]).
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

  // Temporary work vectors. `in_level` is sized once and kept all-false
  // between iterations by undoing exactly the entries it set, rather
  // than re-zeroing all n entries on every level (which is O(n * k)
  // over the whole level structure instead of O(n)).
  std::vector<std::int8_t> in_level(n, false);
  std::vector<int> rv_next;
  std::vector<int> nbr, nbr_next;
  std::vector<int> nrem;

  for (const std::vector<int>& lslevel : ls)
  {
    // Mark all nodes of the current level
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
        std::ranges::sort(nbr, cmp_degree);
        rv.insert(rv.end(), nbr.begin(), nbr.end());
        for (int w : nbr)
          labelled[w] = true;

        // Save nodes for next level to a separate list, rv_next
        std::ranges::sort(nbr_next, cmp_degree);
        rv_next.insert(rv_next.end(), nbr_next.begin(), nbr_next.end());
        for (int w : nbr_next)
          labelled[w] = true;

        ++current_node;
      }

      // Find any remaining unlabelled nodes in level and label the one
      // with lowest degree
      nrem.clear();
      for (int w : lslevel)
      {
        if (!labelled[w])
          nrem.push_back(w);
      }

      if (nrem.empty())
        break;

      std::ranges::sort(nrem, cmp_degree);
      rv.push_back(nrem.front());
      labelled[nrem.front()] = true;
    }

    // Insert already-labelled nodes of next level
    rv.insert(rv.end(), rv_next.begin(), rv_next.end());

    // Undo the marks set for this level so in_level is all-false again
    for (int w : lslevel)
      in_level[w] = false;
  }

  return rv;
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
  // Unlike Gibbs-Poole-Stockmeyer, only the single lowest-degree
  // candidate is tried at each step -- Cuthill-McKee has no
  // width-minimising second phase to feed a wider candidate set into,
  // so testing more candidates here would only add cost, not quality.
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
graph::reorder_gps(const graph::AdjacencyList<std::int32_t>& graph,
                   std::size_t max_candidates, std::size_t num_threads)
{
  const std::int32_t n = graph.num_nodes();
  std::vector<std::int32_t> r(n, -1);
  std::vector<std::int32_t> rv;

  // Repeat for each disconnected part of the graph
  int count = 0;
  while (count < n)
  {
    rv = gps_reorder_unlabelled(graph, r, max_candidates, num_threads);
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
