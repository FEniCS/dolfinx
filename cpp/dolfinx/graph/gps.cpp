// Copyright (C) 2021 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "gps.h"
#include <algorithm>
#include <limits>

#include <dolfinx/common/log.h>

using namespace dolfinx;

namespace
{
// Compute the sets of connected components of the input "graph" which contain
// the nodes in "indices".
std::vector<std::vector<int>>
residual_graph_components(const graph::AdjacencyList<int>& graph,
                          const std::vector<int>& indices)
{
  const int n = graph.num_nodes();
  std::vector<std::vector<int>> rgc;
  if (indices.size() == 0)
    return rgc;

  // Mark all nodes as labelled, except those in the residual graph
  std::vector<bool> labelled(n, true);
  for (int w : indices)
    labelled[w] = false;

  std::vector<int> r;
  // Find first unlabeled entry
  auto it = std::find(labelled.begin(), labelled.end(), false);
  while (it != labelled.end())
  {
    r = {static_cast<int>(std::distance(labelled.begin(), it))};
    labelled[r[0]] = true;

    // Get connected component of graph starting from r[0]
    int c = 0;
    while (c < static_cast<int>(r.size()))
    {
      for (int w : graph.links(r[c]))
      {
        if (labelled[w])
          continue;
        r.push_back(w);
        labelled[w] = true;
      }
      ++c;
    }
    rgc.push_back(r);

    // Find next unlabeled entry
    it = std::find(it, labelled.end(), false);
  }

  std::sort(rgc.begin(), rgc.end(),
            [](const std::vector<int>& a, const std::vector<int>& b) {
              return (a.size() > b.size());
            });

  return rgc;
}

// Get the (maximum) width of a level structure
inline int max_level_width(const std::vector<std::vector<int>>& levels)
{
  return std::max_element(
             levels.begin(), levels.end(),
             [](const std::vector<int>& a, const std::vector<int>& b) {
               return (a.size() < b.size());
             })
      ->size();
}

// Create a level structure from graph, rooted at node s
std::vector<std::vector<int>>
create_level_structure(const graph::AdjacencyList<int>& graph, int s)
{
  const int n = graph.num_nodes();
  std::vector<std::vector<int>> level_structure;
  std::vector<bool> labelled(n, false);
  std::vector<int> level = {s};
  labelled[s] = true;

  while (level.size() > 0)
  {
    level_structure.push_back(level);
    level.clear();
    for (int it : level_structure.back())
    {
      for (int idx : graph.links(it))
      {
        if (idx == it or labelled[idx])
          continue;
        level.push_back(idx);
        labelled[idx] = true;
      }
    }
  }
  return level_structure;
}

} // namespace

std::vector<int> graph::gps_reorder(const graph::AdjacencyList<int>& graph)
{
  const int n = graph.num_nodes();

  // Degree comparison function
  auto cmp_degree = [&graph](int a, int b) {
    return (graph.num_links(a) < graph.num_links(b));
  };

  // ALGORITHM I. Finding endpoints of a pseudo-diameter.

  // A. Pick an arbitrary vertex of minimal degree and call it v
  int dmin = std::numeric_limits<int>::max();
  int v = 0;
  for (int i = 0; i < n; ++i)
  {
    int d = graph.num_links(i);
    if (d < dmin)
    {
      v = i;
      dmin = d;
    }
  }

  // B. Generate a level structure Lv rooted at vertex v.
  std::vector<std::vector<int>> lv = create_level_structure(graph, v);
  std::vector<std::vector<int>> lu;
  bool done = false;
  int u = 0;
  while (!done)
  {
    // Sort final level S of Lv into increasing degree order
    std::vector<int>& S = lv.back();
    std::sort(S.begin(), S.end(), cmp_degree);

    int w_min = std::numeric_limits<int>::max();
    done = true;
    // C. Generate level structures rooted at vertices s in S selected in
    // order of increasing degree.
    for (int s : S)
    {
      std::vector<std::vector<int>> lstmp = create_level_structure(graph, s);
      if (lstmp.size() > lv.size())
      {
        // Found a deeper level structure, so restart
        v = s;
        lv = lstmp;
        done = false;
        break;
      }
      //  D. Let u be the vertex of S whose associated level structure has
      //  smallest width
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

  assert(lv.size() == lu.size());
  int k = lv.size();
  LOG(INFO) << "GPS pseudo-diameter:(" << k << ") " << u << "-" << v << "\n";

  // ALGORITHM II. Minimizing level width.

  // Pair (i, j) associated with each node: lvt=i, lut=j
  std::vector<int> lut(n), lvt(n);
  for (int i = 0; i < k; ++i)
  {
    for (int w : lv[i])
      lvt[w] = i;
    for (int w : lu[i])
      lut[w] = k - 1 - i;
  }

  assert(lut[v] == 0 and lvt[v] == 0);
  assert(lut[u] == (k - 1) and lvt[u] == (k - 1));

  // Insert any nodes (i, i) into new level structure ls and capture residual
  // nodes in rg
  std::vector<std::vector<int>> ls(k);
  std::vector<int> rg;
  for (int i = 0; i < k; ++i)
  {
    for (int w : lu[i])
    {
      if (lut[w] == lvt[w])
        ls[lut[w]].push_back(w);
      else
        rg.push_back(w);
    }
  }

  std::vector<std::vector<int>> rgc = residual_graph_components(graph, rg);

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
      ++wh[lvt[w]];
      ++wl[lut[w]];
    }
    // Zero any entries which did not increase
    std::transform(wh.begin(), wh.end(), wn.begin(), wh.begin(),
                   [](int vh, int vn) { return (vh > vn) ? vh : 0; });
    std::transform(wl.begin(), wl.end(), wn.begin(), wl.begin(),
                   [](int vl, int vn) { return (vl > vn) ? vl : 0; });
    // Find maximum of those that did increase
    int h0 = *std::max_element(wh.begin(), wh.end());
    int l0 = *std::max_element(wl.begin(), wl.end());
    if (h0 < l0)
    {
      for (int w : r)
        ls[lvt[w]].push_back(w);
    }
    else
    {
      for (int w : r)
        ls[lut[w]].push_back(w);
    }
    // TODO: h0 == l0
  }

  // ALGORITHM III. Numbering
  std::vector<int> rv;
  std::vector<bool> labelled(n, false);

  int current_node = 0;
  rv.push_back(v);
  labelled[v] = true;

  // Temporary vectors
  std::vector<bool> in_level;
  std::vector<int> rv_next;
  std::vector<int> nbr, nbr_next;
  std::vector<int> nrem;

  for (std::size_t level = 0; level < ls.size(); ++level)
  {
    // Mark all nodes of the current level
    in_level.assign(n, false);
    for (int w : ls[level])
      in_level[w] = true;

    rv_next.clear();
    while (true)
    {
      while (current_node < (int)rv.size())
      {
        // Get unlabelled neighbours of current node in this level
        // and next level
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
      // Find any remaining unlabelled nodes in level
      // and label the one with lowest degree
      nrem.clear();
      for (int w : ls[level])
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

  if (static_cast<int>(rv.size()) != n)
  {
    throw std::runtime_error(
        "Numbering incomplete: probably disconnected graph");
  }

  // Reverse permutation
  std::vector<int> r(n);
  for (int i = 0; i < n; ++i)
    r[rv[i]] = i;

  return r;
}