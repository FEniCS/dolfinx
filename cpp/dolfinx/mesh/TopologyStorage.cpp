// Copyright (C) 2020 Matthias Rambausek
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "TopologyStorage.h"
#include "PermutationComputation.h"
#include "TopologyComputation.h"

#include <dolfinx/common/IndexMap.h>
#include <dolfinx/graph/AdjacencyList.h>

using namespace dolfinx;
using namespace dolfinx::mesh;
using namespace dolfinx::mesh;

// TopologyStorageLayer free functions
// ===================================
//------------------------------------------------------------------------------
//
std::shared_ptr<const std::vector<bool>> storage::set_interior_facets(
    TopologyStorageLayer& storage,
    std::shared_ptr<const std::vector<bool>> interior_facets)
{
  return storage.interior_facets = interior_facets;
  //         = std::make_shared<const std::vector<bool>>(interior_facets);
}
//------------------------------------------------------------------------------
/// Set connectivity for given pair of topological dimensions in given storage
std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
storage::set_connectivity(
    TopologyStorageLayer& storage, int d0, int d1,
    std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c)
{
  assert(d0 < storage.connectivity.rows());
  assert(d1 < storage.connectivity.cols());
  return storage.connectivity(d0, d1) = c;
}
//------------------------------------------------------------------------------
/// Set index map for entities of dimension dim
std::shared_ptr<const common::IndexMap> storage::set_index_map(
    TopologyStorageLayer& storage,
    int dim, std::shared_ptr<const common::IndexMap> index_map)
{
  assert(dim < static_cast<int>(storage.index_map.size()));
  return storage.index_map[dim] = index_map;
}
//------------------------------------------------------------------------------
std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
storage::set_cell_permutations(
    TopologyStorageLayer& storage,
    std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
        cell_permutations)
{
  return storage.cell_permutations = cell_permutations;
}
//------------------------------------------------------------------------------
std::shared_ptr<
    const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
storage::set_facet_permutations(
    TopologyStorageLayer& storage,
    std::shared_ptr<
        const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
        facet_permutations)
{
  return storage.facet_permutations = facet_permutations;
}
//------------------------------------------------------------------------------
std::shared_ptr<const std::vector<bool>>
storage::interior_facets(const TopologyStorageLayer& storage)
{
  return storage.interior_facets;
}
//------------------------------------------------------------------------------
std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
storage::connectivity(const TopologyStorageLayer& storage, int d0, int d1)
{
  assert(d0 < storage.connectivity.rows());
  assert(d1 < storage.connectivity.cols());
  return storage.connectivity(d0, d1);
}
//------------------------------------------------------------------------------
std::shared_ptr<const common::IndexMap>
storage::index_map(const TopologyStorageLayer& storage, int dim)
{
  assert(dim < static_cast<int>(storage.index_map.size()));
  return storage.index_map[dim];
}
//------------------------------------------------------------------------------
std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
storage::cell_permutations(const TopologyStorageLayer& storage)
{
  return storage.cell_permutations;
}
//------------------------------------------------------------------------------
std::shared_ptr<
    const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
storage::facet_permutations(const TopologyStorageLayer& storage)
{
  return storage.facet_permutations;
}
//------------------------------------------------------------------------------
int storage::assign(TopologyStorageLayer& to, const TopologyStorageLayer& from,
                    bool override)
{
  int count = 0;
  if (auto facets = interior_facets(from);
      facets && (override || !interior_facets(to)))
  {
    set_interior_facets(to, facets);
    ++count;
  }

  for (int jj = 0; jj < from.connectivity.cols(); ++jj)
  {
    if (auto imap = index_map(from, jj);
        imap && (override || !index_map(to, jj)))
    {
      set_index_map(to, jj, imap);
      ++count;
    }
    for (int ii = 0; ii < from.connectivity.rows(); ++ii)
      if (auto conn = connectivity(from, ii, jj);
          conn && (override || !connectivity(to, ii, jj)))
      {
        set_connectivity(to, ii, jj, conn);
        ++count;
      }
  }

  if (auto cpermutations = cell_permutations(from);
      cpermutations && (override || !cell_permutations(to)))
  {
    set_cell_permutations(to, cpermutations);
    ++count;
  }

  if (auto fpermutations = facet_permutations(from);
      fpermutations && (override || !facet_permutations(to)))
  {
    set_facet_permutations(to, fpermutations);
    ++count;
  }

  return count;
}
//------------------------------------------------------------------------------
int storage::assign_where_empty(TopologyStorageLayer& to,
                                const TopologyStorageLayer& from)
{
  return assign(to, from, false);
}
//------------------------------------------------------------------------------
void storage::assign(TopologyStorageLayer& to, const TopologyStorage& from,
                     bool override)
{
  from.visit_from_bottom([&](const TopologyStorageLayer& other) {
    storage::assign(to, other, override);
    // never stop (iterations in visit stop on true)
    return false;
  });
}
//------------------------------------------------------------------------------
void storage::assign(TopologyStorage& to, const TopologyStorage& from,
                     bool override)
{
  to.write([&](TopologyStorageLayer& target) {
    storage::assign(target, from, override);
    return false;
  });
}
//------------------------------------------------------------------------------
void storage::assign_if_not_empty(TopologyStorage& to,
                                  const TopologyStorage& from, bool override)
{
  if (!to.empty())
    assign(to, from, override);
}
