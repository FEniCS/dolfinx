// Copyright (C) 2006-2017 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "TopologyComputation.h"
#include "Cell.h"
#include "CellType.h"
#include "Mesh.h"
#include "MeshConnectivity.h"
#include "MeshIterator.h"
#include "MeshTopology.h"
#include <algorithm>
#include <boost/multi_array.hpp>
#include <boost/unordered_map.hpp>
#include <cstdint>
#include <dolfin/common/Timer.h>
#include <dolfin/common/utils.h>
#include <memory>
#include <spdlog/spdlog.h>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace dolfin;
using namespace dolfin::mesh;

namespace
{

// cell-to-entity (tdim, dim). Ths functions builds a list of all
// entities of Compute mesh entities of given topological dimension, and
// connectivity dimension dim for every cell, keyed by the sorted lists
// of vertex indices that make up the entity. Also attached is whether
// or nor the entity is a ghost, the local entity index (relative to the
// generating cell) and the generating cell index. This list is then
// sorted, with matching keys corresponding to a single entity. The
// entities are numbered such that ghost entities come after al regular
// entities.
//
// Returns the number of entities
//
// The function is templated over the number of vertices that make up an
// entity of dimension dim. This avoid dynamic memory allocations,
// yielding significant performance improvements
template <int N>
std::tuple<std::shared_ptr<MeshConnectivity>, std::shared_ptr<MeshConnectivity>,
           std::int32_t>
compute_entities_by_key_matching(Mesh& mesh, int dim)
{
  // Get mesh topology and connectivity
  MeshTopology& topology = mesh.topology();

  // Check if entities have already been computed
  if (topology.size(dim) > 0)
  {
    // Make sure we really have the connectivity
    if ((!topology.connectivity(topology.dim(), dim)
         && dim != (int)topology.dim())
        || (!topology.connectivity(dim, 0) && dim != 0))
    {
      spdlog::error("TopologyComputation.cpp", "compute topological entities",
                    "Entities of topological dimension %d exist but "
                    "connectivity is missing",
                    dim);
      throw std::runtime_error("Missing connectivity");
    }
    return {nullptr, nullptr, topology.size(dim)};
  }

  // Make sure connectivity does not already exist
  if (topology.connectivity(topology.dim(), dim)
      || topology.connectivity(dim, 0))
  {
    throw std::runtime_error("Connectivity for topological dimension "
                             + std::to_string(dim)
                             + " exists but entities are missing");
  }

  // Start timer
  common::Timer timer("Compute entities dim = " + std::to_string(dim));

  // Get cell type
  const CellType& cell_type = mesh.type();

  // Initialize local array of entities
  const std::int8_t num_entities = cell_type.num_entities(dim);
  const int num_vertices = cell_type.num_vertices(dim);

  // Create map from cell vertices to entity vertices
  boost::multi_array<std::int32_t, 2> e_vertices(
      boost::extents[num_entities][num_vertices]);
  const int num_vertices_per_cell = cell_type.num_vertices();
  std::vector<std::int32_t> v(num_vertices_per_cell);
  std::iota(v.begin(), v.end(), 0);
  cell_type.create_entities(e_vertices, dim, v.data());

  assert(N == num_vertices);

  // Create data structure to hold entities ([vertices key],
  // (cell_local_index, cell index), [entity vertices], entity index)
  std::vector<std::tuple<std::array<std::int32_t, N>,
                         std::pair<std::int8_t, std::int32_t>,
                         std::array<std::int32_t, N>, std::int32_t>>
      keyed_entities(num_entities * mesh.num_cells());

  // Loop over cells to build list of keyed (by vertices) entities
  int entity_counter = 0;
  for (auto& c : MeshRange<Cell>(mesh, MeshRangeType::ALL))
  {
    // Get vertices from cell
    const std::int32_t* vertices = c.entities(0);
    assert(vertices);

    // Iterate over entities of cell
    const int cell_index = c.index();
    for (std::int8_t i = 0; i < num_entities; ++i)
    {
      // Get entity vertices
      auto& entity = std::get<2>(keyed_entities[entity_counter]);
      for (std::int8_t j = 0; j < num_vertices; ++j)
        entity[j] = vertices[e_vertices[i][j]];

      // Sort entity vertices to create key
      auto& entity_key = std::get<0>(keyed_entities[entity_counter]);
      std::partial_sort_copy(entity.begin(), entity.end(), entity_key.begin(),
                             entity_key.end());

      // Attach (local index, cell index), making local_index negative
      // if it is not a ghost cell. This ensures that non-ghosts come
      // before ghosts when sorted. The index is corrected later.
      if (!c.is_ghost())
        std::get<1>(keyed_entities[entity_counter]) = {-i - 1, cell_index};
      else
        std::get<1>(keyed_entities[entity_counter]) = {i, cell_index};

      // Increment entity counter
      ++entity_counter;
    }
  }

  // Sort entities by key. For the same key, those beloning to non-ghost
  // cells will appear before those belonging to ghost cells.
  std::sort(keyed_entities.begin(), keyed_entities.end());

  // Compute entity indices (using -1, -2, -3, etc, for ghost entities)
  std::int32_t nonghost_index(0), ghost_index(-1);
  std::array<std::int32_t, N> previous_key;
  std::fill(previous_key.begin(), previous_key.end(), -1);
  for (auto e = keyed_entities.begin(); e != keyed_entities.end(); ++e)
  {
    const auto& key = std::get<0>(*e);
    if (key == previous_key)
    {
      // Repeated entity, reuse entity index
      std::get<3>(*e) = std::get<3>(*(e - 1));
    }
    else
    {
      // New entity, so give index (negative for ghosts)
      const auto local_index = std::get<1>(*e).first;
      if (local_index < 0)
        std::get<3>(*e) = nonghost_index++;
      else
        std::get<3>(*e) = ghost_index--;

      // Update key
      previous_key = key;
    }

    // Re-map local index (make all positive)
    auto& local_index = std::get<1>(*e).first;
    local_index = (local_index < 0) ? (-local_index - 1) : local_index;
  }

  // Total number of entities
  const std::int32_t num_nonghost_entities = nonghost_index;
  const std::int32_t num_ghost_entities = -(ghost_index + 1);
  const std::int32_t num_mesh_entities
      = num_nonghost_entities + num_ghost_entities;

  // List of vertex indices connected to entity e
  std::vector<std::array<int, N>> connectivity_ev(num_mesh_entities);

  // List of entity e indices connected to cell
  boost::multi_array<int, 2> connectivity_ce(
      boost::extents[mesh.num_cells()][num_entities]);

  // Build connectivity arrays (with ghost entities at the end)
  // std::int32_t previous_index = -1;
  std::int32_t previous_index = std::numeric_limits<std::int32_t>::min();
  for (auto& entity : keyed_entities)
  {
    // Get entity index, and remap for ghosts (negative entity index) to
    // true index
    auto& e_index = std::get<3>(entity);
    if (e_index < 0)
      e_index = num_nonghost_entities - (e_index + 1);

    // Add to enity-to-vertex map if entity is new
    if (e_index != previous_index)
    {
      assert(e_index < (std::int32_t)connectivity_ev.size());
      connectivity_ev[e_index] = std::get<2>(entity);

      // Update index
      previous_index = e_index;
    }

    // Add to cell-to-entity map
    const auto& cell = std::get<1>(entity);
    const auto local_index = cell.first;
    const auto cell_index = cell.second;
    connectivity_ce[cell_index][local_index] = e_index;
  }

  // FIXME: move this out some Mesh can be const

  // Initialise connectivity data structure
  topology.init(dim, num_mesh_entities, 0);

  // Initialise ghost entity offset
  topology.init_ghost(dim, num_nonghost_entities);

  // Set cell-entity connectivity
  auto ce = std::make_shared<MeshConnectivity>(connectivity_ce);
  auto ev = std::make_shared<MeshConnectivity>(connectivity_ev);

  return {ce, ev, connectivity_ev.size()};
}
//-----------------------------------------------------------------------------
// Compute connectivity from transpose
MeshConnectivity compute_from_transpose(const Mesh& mesh, std::size_t d0,
                                        std::size_t d1)
{
  // The transpose is computed in three steps:
  //
  //   1. Iterate over entities of dimension d1 and count the number
  //      of connections for each entity of dimension d0
  //
  //   2. Allocate memory / prepare data structures
  //
  //   3. Iterate again over entities of dimension d1 and add connections
  //      for each entity of dimension d0

  spdlog::info("Computing mesh connectivity %d - %d from transpose.", d0, d1);

  // Get mesh topology and connectivity
  const MeshTopology& topology = mesh.topology();

  // Need connectivity d1 - d0
  if (!topology.connectivity(d1, d0))
    throw std::runtime_error("Missing required connectivity d1-d0.");

  // Temporary array
  std::vector<std::size_t> tmp(topology.size(d0), 0);

  // Count the number of connections
  for (auto& e1 : MeshRange<MeshEntity>(mesh, d1, MeshRangeType::ALL))
    for (auto& e0 : EntityRange<MeshEntity>(e1, d0))
      tmp[e0.index()]++;

  // Initialize the number of connections
  MeshConnectivity connectivity(tmp);

  // Reset current position for each entity
  std::fill(tmp.begin(), tmp.end(), 0);

  // Add the connections
  for (auto& e1 : MeshRange<MeshEntity>(mesh, d1, MeshRangeType::ALL))
    for (auto& e0 : EntityRange<MeshEntity>(e1, d0))
      connectivity.set(e0.index(), e1.index(), tmp[e0.index()]++);

  return connectivity;
}
//-----------------------------------------------------------------------------
// Direct lookup of entity from vertices in a map
MeshConnectivity compute_from_map(const Mesh& mesh, std::size_t d0,
                                  std::size_t d1)
{
  assert(d1 > 0);
  assert(d0 > d1);

  // Get the type of entity d0
  std::unique_ptr<CellType> cell_type(
      CellType::create(mesh.type().entity_type(d0)));

  // Make a map from the sorted d1 entity vertices to the d1 entity index
  boost::unordered_map<std::vector<std::int32_t>, std::int32_t> entity_to_index;
  entity_to_index.reserve(mesh.num_entities(d1));

  const std::size_t num_verts_d1 = mesh.type().num_vertices(d1);
  std::vector<std::int32_t> key(num_verts_d1);
  for (auto& e : MeshRange<MeshEntity>(mesh, d1, MeshRangeType::ALL))
  {
    std::partial_sort_copy(e.entities(0), e.entities(0) + num_verts_d1,
                           key.begin(), key.end());
    entity_to_index.insert({key, e.index()});
  }

  MeshConnectivity connectivity(mesh.num_entities(d0),
                                cell_type->num_entities(d1));

  // Search for d1 entities of d0 in map, and recover index
  std::vector<std::int32_t> entities;
  boost::multi_array<std::int32_t, 2> keys;
  for (auto& e : MeshRange<MeshEntity>(mesh, d0, MeshRangeType::ALL))
  {
    entities.clear();
    cell_type->create_entities(keys, d1, e.entities(0));
    for (const auto& p : keys)
    {
      std::partial_sort_copy(p.begin(), p.end(), key.begin(), key.end());
      const auto it = entity_to_index.find(key);
      assert(it != entity_to_index.end());
      entities.push_back(it->second);
    }
    Eigen::Map<const Eigen::Array<std::int32_t, 1, Eigen::Dynamic>> _e(
        entities.data(), entities.size());
    connectivity.set(e.index(), _e);
  }

  return connectivity;
}
} // namespace

//-----------------------------------------------------------------------------
std::size_t TopologyComputation::compute_entities(Mesh& mesh, std::size_t dim)
{
  spdlog::info("Computing mesh entities of dimension %d", dim);

  // Check if entities have already been computed
  MeshTopology& topology = mesh.topology();
  if (topology.size(dim) > 0)
  {
    // Make sure we really have the connectivity
    if ((!topology.connectivity(topology.dim(), dim) && dim != topology.dim())
        || (!topology.connectivity(dim, 0) && dim != 0))
    {
      spdlog::error("TopologyComputation.cpp", "compute topological entities",
                    "Entities of topological dimension %d exist but "
                    "connectivity is missing",
                    dim);
      throw std::runtime_error("Missing connectivity");
    }
    return topology.size(dim);
  }

  // Call specialised function to compute entities
  const CellType& cell_type = mesh.type();
  const std::int8_t num_entity_vertices = cell_type.num_vertices(dim);
  std::tuple<std::shared_ptr<MeshConnectivity>,
             std::shared_ptr<MeshConnectivity>, std::int32_t>
      data;
  switch (num_entity_vertices)
  {
  case 1:
    data = compute_entities_by_key_matching<1>(mesh, dim);
    break;
  case 2:
    data = compute_entities_by_key_matching<2>(mesh, dim);
    break;
  case 3:
    data = compute_entities_by_key_matching<3>(mesh, dim);
    break;
  case 4:
    data = compute_entities_by_key_matching<4>(mesh, dim);
    break;
  default:
    spdlog::error("TopologyComputation.cpp", "compute topological entities",
                  "Entities with %d vertices not supported",
                  num_entity_vertices);
    throw std::runtime_error("Not supported");
  }

  // Set cell-entity connectivity
  if (std::get<0>(data))
    topology.set_connectivity(std::get<0>(data), topology.dim(), dim);

  // Set entity-vertex connectivity
  if (std::get<1>(data))
    topology.set_connectivity(std::get<1>(data), dim, 0);

  return std::get<2>(data);
}
//-----------------------------------------------------------------------------
void TopologyComputation::compute_connectivity(Mesh& mesh, std::size_t d0,
                                               std::size_t d1)
{
  // This is where all the logic takes place to find a strategy for
  // the connectivity computation. For any given pair (d0, d1), the
  // connectivity is computed by suitably combining the following
  // basic building blocks:
  //
  //   1. compute_entities():     d  - 0  from dim - 0
  //   2. compute_transpose():    d0 - d1 from d1 - d0
  //   3. compute_intersection(): d0 - d1 from d0 - d' - d1
  //   4. compute_from_map():     d0 - d1 from d1 - 0 and d0 - 0
  // Each of these functions assume a set of preconditions that we
  // need to satisfy.

  spdlog::info("Requesting connectivity %d - %d.", d0, d1);

  // Get mesh topology and connectivity
  MeshTopology& topology = mesh.topology();

  // Return connectivity has already been computed
  if (topology.connectivity(d0, d1))
    return;

  // Compute entities if they don't exist
  if (topology.size(d0) == 0)
    compute_entities(mesh, d0);
  if (topology.size(d1) == 0)
    compute_entities(mesh, d1);

  // Check if mesh has entities
  if (topology.size(d0) == 0 && topology.size(d1) == 0)
    return;

  // Check if connectivity still needs to be computed
  if (topology.connectivity(d0, d1))
    return;

  // Start timer
  common::Timer timer("Compute connectivity " + std::to_string(d0) + "-"
                      + std::to_string(d1));

  // Decide how to compute the connectivity
  if (d0 == d1)
  {
    std::vector<std::vector<std::size_t>> connectivity_dd(
        topology.size(d0), std::vector<std::size_t>(1));

    for (auto& e : MeshRange<MeshEntity>(mesh, d0, MeshRangeType::ALL))
      connectivity_dd[e.index()][0] = e.index();
    auto connectivity = std::make_shared<MeshConnectivity>(connectivity_dd);
    topology.set_connectivity(connectivity, d0, d1);
  }
  else if (d0 < d1)
  {
    // Compute connectivity d1 - d0 and take transpose
    compute_connectivity(mesh, d1, d0);
    auto c = std::make_shared<MeshConnectivity>(
        compute_from_transpose(mesh, d0, d1));
    topology.set_connectivity(c, d0, d1);
  }
  else
  {
    // Compute by mapping vertices from a lower dimension entity to
    // those of a higher dimension entity
    auto c = std::make_shared<MeshConnectivity>(compute_from_map(mesh, d0, d1));
    topology.set_connectivity(c, d0, d1);
  }
}
//--------------------------------------------------------------------------
