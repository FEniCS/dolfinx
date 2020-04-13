//
// Created by mrambausek on 12.04.20.
//

#include "TopologyStorage.h"

#include <dolfinx/common/IndexMap.h>
#include <dolfinx/graph/AdjacencyList.h>

using namespace dolfinx;
using namespace dolfinx::mesh;

//------------------------------------------------------------------------------
StorageLock::~StorageLock()
{
  if (!storage.second.expired())
    storage.first->remove_expired_layers();
}
//------------------------------------------------------------------------------
std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
TopologyStorage::connectivity(int d0, int d1) const
{
  for (const auto& layer_pairs : storage_layers)
  {
    // Get the actual storage layer
    const TopologyStorageLayer& layer = layer_pairs.first;
    assert(d0 < layer.connectivity.rows());
    assert(d1 < layer.connectivity.cols());

    // Return if the desired connectivity is present
    if (auto conn = layer.connectivity(d0, d1); conn)
      return conn;
  }
  // Nothing found. Returning empty pointer.
  return std::shared_ptr<const graph::AdjacencyList<std::int32_t>>{};
}
//------------------------------------------------------------------------------
void TopologyStorage::set_connectivity(
    std::shared_ptr<graph::AdjacencyList<std::int32_t>> c, int d0, int d1)
{
  auto layer = active_layer();
  assert(d0 < layer.connectivity.rows());
  assert(d1 < layer.connectivity.cols());
  layer.connectivity(d0, d1) = std::move(c);
}
//------------------------------------------------------------------------------
std::shared_ptr<const common::IndexMap>
TopologyStorage::index_map(int dim) const
{
  for (const auto& layer_pairs : storage_layers)
  {
    // Get the actual storage layer
    const TopologyStorageLayer layer = layer_pairs.first;
    assert(dim < static_cast<int>(layer.index_map.size()));

    // Return if the desired connectivity is present
    if (auto imap = layer.index_map[dim]; imap)
      return imap;
  }
  // Nothing found. Returning empty pointer.
  return std::shared_ptr<const common::IndexMap>{};
}
//------------------------------------------------------------------------------
/// @todo Merge with set_connectivity
///
/// Set the IndexMap for dimension dim
/// @warning This is experimental and likely to change
void TopologyStorage::set_index_map(
    int dim, std::shared_ptr<const common::IndexMap> index_map)
{
  auto layer = active_layer();
  assert(dim < layer.index_map.size());
  layer.index_map[dim] = std::move(index_map);
}
//------------------------------------------------------------------------------
// TODO: make naming consistent with facet permutations
/// Returns the permutation information
std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
TopologyStorage::get_cell_permutation_info() const
{
  for (const auto& layer_pairs : storage_layers)
  {
    // Get the actual storage layer
    const TopologyStorageLayer& layer = layer_pairs.first;

    // Return if the desired connectivity is present
    if (auto permutations = layer.cell_permutations; permutations)
      return permutations;
  }
  // Nothing found. Returning empty pointer.
  return std::shared_ptr<
      const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>{};
}
//------------------------------------------------------------------------------
void TopologyStorage::set_cell_permutations(
    Eigen::Array<std::uint32_t, Eigen::Dynamic, 1> cell_permutations)
{
  active_layer().cell_permutations
      = std::make_shared<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>(
          std::move(cell_permutations));
}
std::shared_ptr<
    const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
TopologyStorage::get_facet_permutations() const
{
  for (const auto& layer_pairs : storage_layers)
  {
    // Get the actual storage layer
    const TopologyStorageLayer& layer = layer_pairs.first;

    // Return if the desired connectivity is present
    if (auto permutations = layer.facet_permutations; permutations)
      return permutations;
  }
  // Nothing found. Returning empty pointer.
  return std::shared_ptr<
      const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>{};
}
//------------------------------------------------------------------------------
void TopologyStorage::set_facet_permutations(
    Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>
        facet_permutations)
{
  active_layer().facet_permutations = std::make_shared<
      const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>(
      std::move(facet_permutations));
}
//------------------------------------------------------------------------------
std::shared_ptr<const std::vector<bool>>
TopologyStorage::interior_facets() const
{
  for (const auto& layer_pairs : storage_layers)
  {
    // Get the actual storage layer
    const TopologyStorageLayer layer = layer_pairs.first;

    // Return if the desired connectivity is present
    if (auto interior_facets = layer.interior_facets; interior_facets)
      return interior_facets;
  }
  // Nothing found. Returning empty pointer.
  return {};
}
//------------------------------------------------------------------------------
void TopologyStorage::set_interior_facets(std::vector<bool> interior_facets)
{
  active_layer().interior_facets
      = std::make_shared<const std::vector<bool>>(std::move(interior_facets));
}
