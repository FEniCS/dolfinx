//
// Created by mrambausek on 12.04.20.
//

#include "TopologyStorage.h"
#include "Partitioning.h"
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
std::shared_ptr<const std::vector<bool>>
storage::set_interior_facets(TopologyStorageLayer& storage,
                    const std::vector<bool>& interior_facets)
{
  return storage.interior_facets
         = std::make_shared<const std::vector<bool>>(interior_facets);
}
//------------------------------------------------------------------------------
/// Set connectivity for given pair of topological dimensions in given storage
std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
storage::set_connectivity(TopologyStorageLayer& storage,
                 std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c,
                 int d0, int d1)
{
  assert(d0 < storage.connectivity.rows());
  assert(d1 < storage.connectivity.cols());
  return storage.connectivity(d0, d1) = c;
}
//------------------------------------------------------------------------------
/// Set index map for entities of dimension dim
std::shared_ptr<const common::IndexMap>
storage::set_index_map(TopologyStorageLayer& storage,
              std::shared_ptr<const common::IndexMap> index_map, int dim)
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
//
//
//
//// TopologyStorageManager free functions
//// =====================================
////------------------------------------------------------------------------------
std::shared_ptr<const std::vector<bool>>
storage::set_interior_facets(TopologyStorageManager& storage,
                    std::shared_ptr<const std::vector<bool>> interior_facets)
{
  return storage.write([&](TopologyStorageLayer& layer) {return set_interior_facets(layer, interior_facets);});
}
//------------------------------------------------------------------------------
/// Set connectivity for given pair of topological dimensions in given storage
std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
storage::set_connectivity(TopologyStorageManager& storage,
                 std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c,
                 int d0, int d1)
{
  return storage.write([&](TopologyStorageLayer& layer) {return set_connectivity(layer, c, d0, d1);});
}
//------------------------------------------------------------------------------
/// Set index map for entities of dimension dim
std::shared_ptr<const common::IndexMap>
storage::set_index_map(TopologyStorageManager& storage,
              std::shared_ptr<const common::IndexMap> index_map, int dim)
{
  return storage.write([&](TopologyStorageLayer& layer) {return set_index_map(layer, index_map, dim);});
}
//------------------------------------------------------------------------------
std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
storage::set_cell_permutations(
    TopologyStorageManager& storage,
    std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
    cell_permutations)
{
  return storage.write([&](TopologyStorageLayer& layer) {return set_cell_permutations(layer, cell_permutations);});

}
//------------------------------------------------------------------------------
std::shared_ptr<
    const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
storage::set_facet_permutations(
    TopologyStorageManager& storage,
    std::shared_ptr<
        const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
    facet_permutations)
{
  return storage.write([&](TopologyStorageLayer& layer) {return set_facet_permutations(layer, facet_permutations);});

}
//------------------------------------------------------------------------------
std::shared_ptr<const std::vector<bool>>
storage::interior_facets(const TopologyStorageManager& storage)
{
  return storage.read([&](const TopologyStorageLayer& layer) {return interior_facets(layer);});
}
//------------------------------------------------------------------------------
std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
storage::connectivity(const TopologyStorageManager& storage, int d0, int d1)
{
  return storage.read([&](const TopologyStorageLayer& layer) {return connectivity(layer, d0, d1);});

}
//------------------------------------------------------------------------------
std::shared_ptr<const common::IndexMap>
storage::index_map(const TopologyStorageManager& storage, int dim)
{
  return storage.read([&](const TopologyStorageLayer& layer) {return index_map(layer, dim);});
}
//------------------------------------------------------------------------------
std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
storage::cell_permutations(const TopologyStorageManager& storage)
{
  return storage.read([&](const TopologyStorageLayer& layer) {return cell_permutations(layer);});
}
//------------------------------------------------------------------------------
std::shared_ptr<
    const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
storage::facet_permutations(const TopologyStorageManager& storage)
{
  return storage.read([&](const TopologyStorageLayer& layer) {
    return facet_permutations(layer);
  });
}
