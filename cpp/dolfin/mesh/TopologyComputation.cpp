// Copyright (C) 2006-2017 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "TopologyComputation.h"
#include "Cell.h"
#include "CellType.h"
#include "Connectivity.h"
#include "Mesh.h"
#include "MeshIterator.h"
#include "Topology.h"
#include "utils.h"
#include <Eigen/Dense>
#include <algorithm>
#include <boost/unordered_map.hpp>
#include <cstdint>
#include <dolfin/common/Timer.h>
#include <dolfin/common/log.h>
#include <dolfin/common/utils.h>
#include <memory>
#include <numeric>
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
std::tuple<std::shared_ptr<Connectivity>, std::shared_ptr<Connectivity>,
           std::int32_t>
compute_entities_by_key_matching(const Mesh& mesh, int dim)
{
  if (dim == 0)
  {
    throw std::runtime_error(
        "Cannot create vertices fo topology. Should already exist.");
  }

  // Get mesh topology and connectivity
  const Topology& topology = mesh.topology();
  const int tdim = topology.dim();

  // Check if entities have already been computed
  if (topology.connectivity(dim, 0))
  {
    // Check that we have cell-entity connectivity
    if (!topology.connectivity(tdim, dim))
      throw std::runtime_error("Missing cell-entity connectivity");

    return {nullptr, nullptr, topology.size(dim)};
  }

  // Start timer
  common::Timer timer("Compute entities of dim = " + std::to_string(dim));

  // Get cell type
  const mesh::CellTypeOld& cell_type = mesh.type();

  // Initialize local array of entities
  const std::int8_t num_entities = mesh::cell_num_entities(cell_type.type, dim);
  const int num_vertices
      = mesh::num_cell_vertices(mesh::cell_entity_type(cell_type.type, dim));

  // Create map from cell vertices to entity vertices
  Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      e_vertices(num_entities, num_vertices);
  const int num_vertices_per_cell = mesh::num_cell_vertices(cell_type.type);
  std::vector<std::int32_t> v(num_vertices_per_cell);
  std::iota(v.begin(), v.end(), 0);
  // cell_type.create_entities(e_vertices, dim, v.data());
  mesh::create_entities(e_vertices, dim, v.data(), cell_type.type);

  assert(N == num_vertices);

  // Create data structure to hold entities ([vertices key],
  // (cell_local_index, cell index), [entity vertices], entity index)
  std::vector<std::tuple<std::array<std::int32_t, N>,
                         std::pair<std::int8_t, std::int32_t>,
                         std::array<std::int32_t, N>, std::int32_t>>
      keyed_entities(num_entities * mesh.num_entities(tdim));

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
        entity[j] = vertices[e_vertices(i, j)];

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

  // Sort entities by key. For the same key, those belonging to
  // non-ghost cells will appear before those belonging to ghost cells.
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
  Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      connectivity_ce(mesh.num_entities(tdim), num_entities);

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
    connectivity_ce(cell_index, local_index) = e_index;
  }

  // FIXME: move this out some Mesh can be const

  // Set cell-entity connectivity
  auto ce = std::make_shared<Connectivity>(connectivity_ce);
  auto ev = std::make_shared<Connectivity>(connectivity_ev);

  return {ce, ev, num_nonghost_entities};
}
//-----------------------------------------------------------------------------
// Compute connectivity from transpose
Connectivity compute_from_transpose(const Mesh& mesh, int d0, int d1)
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

  LOG(INFO) << "Computing mesh connectivity " << d0 << " - " << d1
            << "from transpose.";

  // Get mesh topology and connectivity
  const Topology& topology = mesh.topology();

  // Need connectivity d1 - d0
  if (!topology.connectivity(d1, d0))
    throw std::runtime_error("Missing required connectivity d1-d0.");

  // Compute number of connections for each e0
  std::vector<std::int32_t> num_connections(topology.size(d0), 0);
  for (auto& e1 : MeshRange<MeshEntity>(mesh, d1, MeshRangeType::ALL))
    for (auto& e0 : EntityRange<MeshEntity>(e1, d0))
      num_connections[e0.index()]++;

  // Compute offsets
  std::vector<std::int32_t> offsets(num_connections.size() + 1, 0);
  std::partial_sum(num_connections.begin(), num_connections.end(),
                   offsets.begin() + 1);

  std::vector<std::int32_t> counter(num_connections.size(), 0);
  std::vector<std::int32_t> connections(offsets.back());
  for (auto& e1 : MeshRange<MeshEntity>(mesh, d1, MeshRangeType::ALL))
    for (auto& e0 : EntityRange<MeshEntity>(e1, d0))
      connections[offsets[e0.index()] + counter[e0.index()]++] = e1.index();

  return Connectivity(connections, offsets);
}
//-----------------------------------------------------------------------------
// Direct lookup of entity from vertices in a map
Connectivity compute_from_map(const Mesh& mesh, int d0, int d1)
{
  assert(d1 > 0);
  assert(d0 > d1);

  // Get the type of entity d0
  std::unique_ptr<mesh::CellTypeOld> cell_type(
      mesh::CellTypeOld::create(mesh::cell_entity_type(mesh.type().type, d0)));

  // Make a map from the sorted d1 entity vertices to the d1 entity index
  boost::unordered_map<std::vector<std::int32_t>, std::int32_t> entity_to_index;
  entity_to_index.reserve(mesh.num_entities(d1));

  const std::size_t num_verts_d1
      = mesh::num_cell_vertices(mesh::cell_entity_type(mesh.type().type, d1));

  std::vector<std::int32_t> key(num_verts_d1);
  for (auto& e : MeshRange<MeshEntity>(mesh, d1, MeshRangeType::ALL))
  {
    std::partial_sort_copy(e.entities(0), e.entities(0) + num_verts_d1,
                           key.begin(), key.end());
    entity_to_index.insert({key, e.index()});
  }

  Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      connections(mesh.num_entities(d0),
                  mesh::cell_num_entities(cell_type->type, d1));

  // Search for d1 entities of d0 in map, and recover index
  std::vector<std::int32_t> entities;
  Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      keys;
  for (auto& e : MeshRange<MeshEntity>(mesh, d0, MeshRangeType::ALL))
  {
    entities.clear();
    // cell_type->create_entities(keys, d1, e.entities(0));
    mesh::create_entities(keys, d1, e.entities(0), cell_type->type);
    for (Eigen::Index i = 0; i < keys.rows(); ++i)
    {
      std::partial_sort_copy(keys.row(i).data(),
                             keys.row(i).data() + keys.row(i).cols(),
                             key.begin(), key.end());
      const auto it = entity_to_index.find(key);
      assert(it != entity_to_index.end());
      entities.push_back(it->second);
    }
    for (std::size_t k = 0; k < entities.size(); ++k)
      connections(e.index(), k) = entities[k];
  }

  return Connectivity(connections);
}
} // namespace

//-----------------------------------------------------------------------------
void TopologyComputation::compute_entities(Mesh& mesh, int dim)
{
  LOG(INFO) << "Computing mesh entities of dimension " << dim;

  // Check if entities have already been computed
  Topology& topology = mesh.topology();

  // Vertices must always exist
  if (dim == 0)
    return;

  if (topology.connectivity(dim, 0))
  {
    // Make sure we really have the connectivity
    if (!topology.connectivity(topology.dim(), dim))
    {
      throw std::runtime_error(
          "Cannot compute topological entities. Entities of topological "
          "dimension "
          + std::to_string(dim) + " exist but connectivity is missing.");
    }
    return;
  }

  // Call specialised function to compute entities
  const std::int8_t num_entity_vertices
      = mesh::num_cell_vertices(mesh::cell_entity_type(mesh.type().type, dim));

  std::tuple<std::shared_ptr<Connectivity>, std::shared_ptr<Connectivity>,
             std::int32_t>
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
    throw std::runtime_error("Topology computation of entities with "
                             + std::to_string(num_entity_vertices)
                             + "not supported");
  }
  // Set cell-entity connectivity
  if (std::get<0>(data))
    topology.set_connectivity(std::get<0>(data), topology.dim(), dim);

  // Set entity-vertex connectivity
  if (std::get<1>(data))
    topology.set_connectivity(std::get<1>(data), dim, 0);

  // Initialise ghost entity offset
  topology.init_ghost(dim, std::get<2>(data));
}
//-----------------------------------------------------------------------------
void TopologyComputation::compute_connectivity(Mesh& mesh, int d0, int d1)
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

  LOG(INFO) << "Requesting connectivity " << d0 << " - " << d1;

  // Get mesh topology and connectivity
  Topology& topology = mesh.topology();

  // Return connectivity has already been computed
  if (topology.connectivity(d0, d1))
    return;

  // Compute entities if they don't exist
  if (!topology.connectivity(d0, 0))
    compute_entities(mesh, d0);
  if (!topology.connectivity(d1, 0))
    compute_entities(mesh, d1);

  // Check if connectivity still needs to be computed
  if (topology.connectivity(d0, d1))
    return;

  // Start timer
  common::Timer timer("Compute connectivity " + std::to_string(d0) + "-"
                      + std::to_string(d1));

  // Decide how to compute the connectivity
  if (d0 == d1)
  {
    // For d0-d1, use indentity connecticity
    std::vector<std::vector<std::size_t>> connectivity_dd(
        topology.size(d0), std::vector<std::size_t>(1));
    for (auto& e : MeshRange<MeshEntity>(mesh, d0, MeshRangeType::ALL))
      connectivity_dd[e.index()][0] = e.index();
    auto connectivity = std::make_shared<Connectivity>(connectivity_dd);
    topology.set_connectivity(connectivity, d0, d1);
  }
  else if (d0 < d1)
  {
    // Compute connectivity d1 - d0 and take transpose
    compute_connectivity(mesh, d1, d0);
    auto c
        = std::make_shared<Connectivity>(compute_from_transpose(mesh, d0, d1));
    topology.set_connectivity(c, d0, d1);
  }
  else if (d0 > d1)
  {
    // Compute by mapping vertices from a lower dimension entity to
    // those of a higher dimension entity
    auto c = std::make_shared<Connectivity>(compute_from_map(mesh, d0, d1));
    topology.set_connectivity(c, d0, d1);
  }
  else
    throw std::runtime_error("Entity dimension error when computing topology.");
}
//--------------------------------------------------------------------------
