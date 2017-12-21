// Copyright (C) 2006-2017 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

#include <algorithm>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <boost/multi_array.hpp>
#include <boost/unordered_map.hpp>
#include <boost/version.hpp>

#include <dolfin/common/Timer.h>
#include <dolfin/common/utils.h>
#include <dolfin/log/log.h>
#include "Cell.h"
#include "CellType.h"
#include "Mesh.h"
#include "MeshConnectivity.h"
#include "MeshEntityIterator.h"
#include "MeshTopology.h"
#include "TopologyComputation.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::size_t TopologyComputation::compute_entities(Mesh& mesh, std::size_t dim)
{
  log(TRACE, "Computing mesh entities of dimension %d", dim);

  // Check if entities have already been computed
  const MeshTopology& topology = mesh.topology();
  const MeshConnectivity& ce = topology(topology.dim(), dim);
  const MeshConnectivity& ev = topology(dim, 0);
  if (topology.size(dim) > 0)
  {
    // Make sure we really have the connectivity
    if ((ce.empty() && dim != topology.dim()) || (ev.empty() && dim != 0))
    {
      dolfin_error("TopologyComputation.cpp",
                   "compute topological entities",
                   "Entities of topological dimension %d exist but connectivity is missing", dim);
    }
    return topology.size(dim);
  }

  // Call specialised function to compute entities
  const CellType& cell_type = mesh.type();
  const std::int8_t num_entity_vertices = cell_type.num_vertices(dim);
  switch (num_entity_vertices)
  {
  case  1:
    return TopologyComputation::compute_entities_by_key_matching<1>(mesh, dim);
  case  2:
    return TopologyComputation::compute_entities_by_key_matching<2>(mesh, dim);
  case  3:
    return TopologyComputation::compute_entities_by_key_matching<3>(mesh, dim);
  case  4:
    return TopologyComputation::compute_entities_by_key_matching<4>(mesh, dim);
  default:
    dolfin_error("TopologyComputation.cpp",
                 "compute topological entities",
                 "Entities with %d vertices not supported",
                 num_entity_vertices);
    return 0;
  }
}
//-----------------------------------------------------------------------------
void TopologyComputation::compute_connectivity(Mesh& mesh,
                                               std::size_t d0,
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

  log(TRACE, "Requesting connectivity %d - %d.", d0, d1);

  // Get mesh topology and connectivity
  MeshTopology& topology = mesh.topology();
  MeshConnectivity& connectivity = topology(d0, d1);

  // Check if connectivity has already been computed
  if (!connectivity.empty())
    return;

  // Compute entities if they don't exist
  if (topology.size(d0) == 0)
    compute_entities(mesh, d0);
  if (topology.size(d1) == 0)
    compute_entities(mesh, d1);

  // Check is mesh has entities
  if (topology.size(d0) == 0 && topology.size(d1) == 0)
    return;

  // Check if connectivity still needs to be computed
  if (!connectivity.empty())
    return;

  // Start timer
  Timer timer("Compute connectivity " + std::to_string(d0) + "-"
              + std::to_string(d1));

  // Decide how to compute the connectivity
  if (d0 == d1)
  {
    std::vector<std::vector<std::size_t>>
      connectivity_dd(topology.size(d0), std::vector<std::size_t>(1));
    for (MeshEntityIterator e(mesh, d0, "all"); !e.end(); ++e)
      connectivity_dd[e->index()][0] = e->index();
    topology(d0, d0).set(connectivity_dd);
  }
  else if (d0 < d1)
  {
    // Compute connectivity d1 - d0 and take transpose
    compute_connectivity(mesh, d1, d0);
    compute_from_transpose(mesh, d0, d1);
  }
  else
  {
    // Compute by mapping vertices from a lower dimension entity
    // to those of a higher dimension entity
    compute_from_map(mesh, d0, d1);
  }
}
//--------------------------------------------------------------------------
template<int N>
std::int32_t TopologyComputation::compute_entities_by_key_matching(Mesh& mesh,
                                                                   int dim)
{
  // Get mesh topology and connectivity
  MeshTopology& topology = mesh.topology();
  MeshConnectivity& ce = topology(topology.dim(), dim);
  MeshConnectivity& ev = topology(dim, 0);

  // Check if entities have already been computed
  if (topology.size(dim) > 0)
  {
    // Make sure we really have the connectivity
    if ((ce.empty() && dim != (int) topology.dim()) || (ev.empty() && dim != 0))
    {
      dolfin_error("TopologyComputation.cpp",
                   "compute topological entities",
                   "Entities of topological dimension %d exist but connectivity is missing", dim);
    }
    return topology.size(dim);
  }

  // Make sure connectivity does not already exist
  if (!ce.empty() || !ev.empty())
  {
    dolfin_error("TopologyComputation.cpp",
                 "compute topological entities",
                 "Connectivity for topological dimension %d exists but entities are missing", dim);
  }

  // Start timer
  Timer timer("Compute entities dim = " + std::to_string(dim));

  // Get cell type
  const CellType& cell_type = mesh.type();

  // Initialize local array of entities
  const std::int8_t num_entities = cell_type.num_entities(dim);
  const int num_vertices = cell_type.num_vertices(dim);

  // Create map from cell vertices to entity vertices
  boost::multi_array<unsigned int, 2>
    e_vertices(boost::extents[num_entities][num_vertices]);
  const int num_vertices_per_cell = cell_type.num_vertices();
  std::vector<unsigned int> v(num_vertices_per_cell);
  std::iota(v.begin(), v.end(), 0);
  cell_type.create_entities(e_vertices, dim, v.data());

  dolfin_assert(N == num_vertices);

  // Create data structure to hold entities
  // ([vertices key], (cell_local_index, cell index), [entity vertices], entity index)
  std::vector<std::tuple<std::array<std::int32_t, N>,
                         std::pair<std::int8_t, std::int32_t>,
                         std::array<std::int32_t, N>, std::int32_t>>
    keyed_entities(num_entities*mesh.num_cells());

  // Loop over cells to build list of keyed (by vertices) entities
  int entity_counter = 0;
  for (CellIterator c(mesh, "all"); !c.end(); ++c)
  {
    // Get vertices from cell
    const unsigned int* vertices = c->entities(0);
    dolfin_assert(vertices);

    // Iterate over entities of cell
    const int cell_index = c->index();
    for (std::int8_t i = 0; i < num_entities; ++i)
    {
      // Get entity vertices
      auto& entity = std::get<2>(keyed_entities[entity_counter]);
      for (std::int8_t j = 0; j < num_vertices; ++j)
        entity[j] = vertices[e_vertices[i][j]];

      // Sort entity vertices to create key
      auto& entity_key = std::get<0>(keyed_entities[entity_counter]);
      std::partial_sort_copy(entity.begin(), entity.end(),
                             entity_key.begin(), entity_key.end());

      // Attach (local index, cell index), making local_index negative
      // if it is not a ghost cell. This ensures that non-ghosts come
      // before ghosts when sorted. The index is corrected later.
      if (!c->is_ghost())
        std::get<1>(keyed_entities[entity_counter]) = {-i - 1, cell_index};
      else
        std::get<1>(keyed_entities[entity_counter]) = {i, cell_index};

      // Increment entity counter
      ++entity_counter;
    }
  }

  // Sort entities by key. For the same key, those beloning to
  // non-ghost cells will appear before those belonging to ghost
  // cells.
  std::sort(keyed_entities.begin(), keyed_entities.end());

  // Compute entity indices (using -1, -2, -3, etc, for ghost
  // entities)
  std::int32_t nonghost_index(0), ghost_index(-1);
  std::array<std::int32_t, N> previous_key;
  std::fill(previous_key.begin(), previous_key.end(), -1);
  for (auto e = keyed_entities.begin(); e!= keyed_entities.end(); ++e)
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
  const std::int32_t num_mesh_entities = num_nonghost_entities + num_ghost_entities;

  // List of vertex indices connected to entity e
  std::vector<std::array<int, N>> connectivity_ev(num_mesh_entities) ;

  // List of entity e indices connected to cell
  boost::multi_array<int, 2>
    connectivity_ce(boost::extents[mesh.num_cells()][num_entities]);

  // Build connectivity arrays (with ghost entities at the end)
  //std::int32_t previous_index = -1;
  std::int32_t previous_index = std::numeric_limits<std::int32_t>::min();
  for (auto& entity : keyed_entities)
  {
    // Get entity index, and remap for ghosts (negative entity index)
    // to true index
    auto& e_index = std::get<3>(entity);
    if (e_index < 0)
      e_index = num_nonghost_entities - (e_index + 1);

    // Add to enity-to-vertex map if entity is new
    if (e_index != previous_index)
    {
      dolfin_assert(e_index < (std::int32_t) connectivity_ev.size());
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

  // Initialise connectivity data structure
  topology.init(dim, num_mesh_entities, 0);

  // Initialise ghost entity offset
  topology.init_ghost(dim, num_nonghost_entities);

  // Set cell-entity connectivity
  ce.set(connectivity_ce);
  ev.set(connectivity_ev);

  return connectivity_ev.size();
}
//-----------------------------------------------------------------------------
void TopologyComputation::compute_from_transpose(Mesh& mesh, std::size_t d0,
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

  log(TRACE, "Computing mesh connectivity %d - %d from transpose.", d0, d1);

  // Get mesh topology and connectivity
  MeshTopology& topology = mesh.topology();
  MeshConnectivity& connectivity = topology(d0, d1);

  // Need connectivity d1 - d0
  dolfin_assert(!topology(d1, d0).empty());

  // Temporary array
  std::vector<std::size_t> tmp(topology.size(d0), 0);

  // Count the number of connections
  for (MeshEntityIterator e1(mesh, d1, "all"); !e1.end(); ++e1)
    for (MeshEntityIterator e0(*e1, d0); !e0.end(); ++e0)
      tmp[e0->index()]++;

  // Initialize the number of connections
  connectivity.init(tmp);

  // Reset current position for each entity
  std::fill(tmp.begin(), tmp.end(), 0);

  // Add the connections
  for (MeshEntityIterator e1(mesh, d1, "all"); !e1.end(); ++e1)
    for (MeshEntityIterator e0(*e1, d0); !e0.end(); ++e0)
      connectivity.set(e0->index(), e1->index(), tmp[e0->index()]++);
}
//----------------------------------------------------------------------------
void TopologyComputation::compute_from_map(Mesh& mesh,
                                           std::size_t d0,
                                           std::size_t d1)
{
  dolfin_assert(d1 > 0);
  dolfin_assert(d0 > d1);

  // Get the type of entity d0
  std::unique_ptr<CellType> cell_type(CellType::create(mesh.type()
                                                       .entity_type(d0)));

  MeshConnectivity& connectivity = mesh.topology()(d0, d1);
  connectivity.init(mesh.num_entities(d0), cell_type->num_entities(d1));

  // Make a map from the sorted d1 entity vertices to the d1 entity index
  boost::unordered_map<std::vector<unsigned int>, unsigned int>
    entity_to_index;
  entity_to_index.reserve(mesh.num_entities(d1));

  const std::size_t num_verts_d1 = mesh.type().num_vertices(d1);
  std::vector<unsigned int> key(num_verts_d1);
  for (MeshEntityIterator e(mesh, d1, "all"); !e.end(); ++e)
  {
    std::partial_sort_copy(e->entities(0), e->entities(0) + num_verts_d1,
                           key.begin(), key.end());
    entity_to_index.insert({key, e->index()});
  }

  // Search for d1 entities of d0 in map, and recover index
  std::vector<std::size_t> entities;
  boost::multi_array<unsigned int, 2> keys;
  for (MeshEntityIterator e(mesh, d0, "all"); !e.end(); ++e)
  {
    entities.clear();
    cell_type->create_entities(keys, d1, e->entities(0));
    for (const auto &p : keys)
    {
      std::partial_sort_copy(p.begin(), p.end(), key.begin(), key.end());
      const auto it = entity_to_index.find(key);
      dolfin_assert(it != entity_to_index.end());
      entities.push_back(it->second);
    }
    connectivity.set(e->index(), entities.data());
  }

}
//-----------------------------------------------------------------------------
void TopologyComputation::compute_from_intersection(Mesh& mesh,
                                                    std::size_t d0,
                                                    std::size_t d1,
                                                    std::size_t d)
{
  log(TRACE, "Computing mesh connectivity %d - %d from intersection %d - %d - %d.",
      d0, d1, d0, d, d1);

  // Get mesh topology
  MeshTopology& topology = mesh.topology();

  // Check preconditions
  dolfin_assert(d0 >= d1);
  dolfin_assert(!topology(d0, d).empty());
  dolfin_assert(!topology(d, d1).empty());

  // Temporary dynamic storage, later copied into static storage
  std::vector<std::vector<std::size_t>> connectivity(topology.size(d0));

  // A bitmap used to ensure we do not store duplicates
  std::vector<bool> e1_visited(topology.size(d1));

  // Iterate over all entities of dimension d0
  std::size_t max_size = 1;
  const std::size_t e0_num_entities = mesh.type().num_vertices(d0);
  const std::size_t e1_num_entities = mesh.type().num_vertices(d1);
  std::vector<std::size_t> _e0(e0_num_entities);
  std::vector<std::size_t> _e1(e1_num_entities);
  for (MeshEntityIterator e0(mesh, d0, "all"); !e0.end(); ++e0)
  {
    // Get set of connected entities for current entity
    std::vector<std::size_t>& entities = connectivity[e0->index()];

    // Reserve space
    entities.reserve(max_size);

    // Sorted list of e0 vertex indices (necessary to test for
    // presence of one list in another)
    std::copy(e0->entities(0), e0->entities(0) + e0_num_entities, _e0.begin());
    std::sort(_e0.begin(), _e0.end());

    // Initialise e1_visited to false for all neighbours of e0. The
    // loop structure mirrors the one below.
    for (MeshEntityIterator e(*e0, d); !e.end(); ++e)
      for (MeshEntityIterator e1(*e, d1); !e1.end(); ++e1)
        e1_visited[e1->index()] = false;

    // Iterate over all connected entities of dimension d
    for (MeshEntityIterator e(*e0, d); !e.end(); ++e)
    {
      // Iterate over all connected entities of dimension d1
      for (MeshEntityIterator e1(*e, d1); !e1.end(); ++e1)
      {
        // Skip already visited connected entities (to avoid duplicates)
        if (e1_visited[e1->index()])
          continue;
        e1_visited[e1->index()] = true;

        if (d0 == d1)
        {
          // An entity is not a neighbor to itself (duplicate index
          // entries removed at end)
          if (e0->index() != e1->index())
            entities.push_back(e1->index());
        }
        else
        {
          // Sorted list of e1 vertex indices
          std::copy(e1->entities(0), e1->entities(0) + e1_num_entities,
                    _e1.begin());
          std::sort(_e1.begin(), _e1.end());

          // Entity e1 must be completely contained in e0
          if (std::includes(_e0.begin(), _e0.end(), _e1.begin(), _e1.end()))
            entities.push_back(e1->index());
        }
      }
    }

    // Store maximum size
    max_size = std::max(entities.size(), max_size);
  }

  // Copy to static storage
  topology(d0, d1).set(connectivity);
}
//-----------------------------------------------------------------------------
