// Copyright (C) 2006-2014 Anders Logg and Garth N. Wells
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
//
// Modified by Garth N. Wells 2012.
//
// First added:  2006-06-02
// Last changed: 2014-01-17

#include <algorithm>
#include <vector>
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
  // Get mesh topology and connectivity
  MeshTopology& topology = mesh.topology();
  MeshConnectivity& ce = topology(topology.dim(), dim);
  MeshConnectivity& ev = topology(dim, 0);

  // Check if entities have already been computed
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

  // Make sure connectivity does not already exist
  if (!ce.empty() || !ev.empty())
  {
    dolfin_error("TopologyComputation.cpp",
                 "compute topological entities",
                 "Connectivity for topological dimension %d exists but entities are missing", dim);
  }


  // Start timer
  Timer timer("compute entities dim = " + to_string(dim));

  // Get cell type
  const CellType& cell_type = mesh.type();

  // Initialize local array of entities
  const std::size_t m = cell_type.num_entities(dim);
  const std::size_t n = cell_type.num_vertices(dim);
  std::vector<std::vector<unsigned int> >
    e_vertices(m, std::vector<unsigned int>(n, 0));

  // List of entity e indices connected to cell
  std::vector<std::vector<unsigned int> > connectivity_ce(mesh.num_cells());

  // List of vertces indices connected to entity e
  std::vector<std::vector<unsigned int> > connectivity_ev;

  std::size_t current_entity = 0;
  std::size_t max_ce_connections = 1;
  boost::unordered_map<std::vector<unsigned int>, unsigned int>
    evertices_to_index;

  // Rehash/reserve map for efficiency
  const std::size_t max_elements
    = mesh.num_cells()*mesh.type().num_entities(dim)/2;
  #if BOOST_VERSION < 105000
  evertices_to_index.rehash(max_elements/evertices_to_index.max_load_factor()
                            + 1);
  #else
  evertices_to_index.reserve(max_elements);
  #endif

  // Loop over cells
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    // Cell index
    const std::size_t cell_index = c->index();

    // Reserve space to reduce dynamic allocations
    connectivity_ce[cell_index].reserve(max_ce_connections);

    // Get vertices from cell
    const unsigned int* vertices = c->entities(0);
    dolfin_assert(vertices);

    // Create entities
    cell_type.create_entities(e_vertices, dim, vertices);

    // Iterate over the given list of entities
    std::vector<std::vector<unsigned int> >::iterator entity;
    for (entity = e_vertices.begin(); entity != e_vertices.end(); ++entity)
    {
      // Sort entities (to use as map key)
      std::sort(entity->begin(), entity->end());

      // Insert into map
      std::pair<boost::unordered_map<std::vector<unsigned int>, unsigned int>::iterator, bool>
        it = evertices_to_index.insert(std::make_pair(*entity, current_entity));

      // Entity index
      std::size_t e_index = it.first->second;

      // Add entity index to cell - e connectivity
      connectivity_ce[cell_index].push_back(e_index);

      // If new key was inserted, increment entity counter
      if (it.second)
      {
        // Add list of new entity vertices
        connectivity_ev.push_back(*entity);

        // Update max vector size (used to reserve space for performance);
        max_ce_connections = std::max(max_ce_connections,
                                      connectivity_ce[cell_index].size());

        // Increase counter
        current_entity++;
      }
    }
  }

  // Initialise connectivity data structure
  topology.init(dim, connectivity_ev.size(), connectivity_ev.size());

  // Copy connectivity data into static MeshTopology data structures
  ce.set(connectivity_ce);
  ev.set(connectivity_ev);

  return current_entity;
}
//-----------------------------------------------------------------------------
void TopologyComputation::compute_connectivity(Mesh& mesh,
                                               std::size_t d0,
                                               std::size_t d1)
{
  // This is where all the logic takes place to find a stragety for
  // the connectivity computation. For any given pair (d0, d1), the
  // connectivity is computed by suitably combining the following
  // basic building blocks:
  //
  //   1. compute_entities():     d  - 0  from dim - 0
  //   2. compute_transpose():    d0 - d1 from d1 - d0
  //   3. compute_intersection(): d0 - d1 from d0 - d' - d1
  //
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
  Timer timer("compute connectivity " + to_string(d0) + " - " + to_string(d1));

  // Decide how to compute the connectivity
  if (d0 == 0 && d1 == 0)
  {
    std::vector<std::vector<std::size_t> >
      connectivity00(topology.size(d0), std::vector<std::size_t>(1));
    for (MeshEntityIterator v(mesh, d0); !v.end(); ++v)
      connectivity00[v->index()][0] = v->index();
    topology(d0, d0).set(connectivity00);
  }
  else if (d0 < d1)
  {
    // Compute connectivity d1 - d0 and take transpose
    compute_connectivity(mesh, d1, d0);
    compute_from_transpose(mesh, d0, d1);
  }
  else
  {
    // These connections should already exist
    dolfin_assert(!(d0 > 0 && d1 == 0));

    // Choose how to take intersection
    dolfin_assert(d0 != 0);
    dolfin_assert(d1 != 0);
    std::size_t d = 0;

    // Compute connectivity d0 - d - d1 and take intersection
    compute_connectivity(mesh, d0, d);
    compute_connectivity(mesh, d, d1);
    compute_from_intersection(mesh, d0, d1, d);
  }
}
//--------------------------------------------------------------------------
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
  for (MeshEntityIterator e1(mesh, d1); !e1.end(); ++e1)
    for (MeshEntityIterator e0(*e1, d0); !e0.end(); ++e0)
      tmp[e0->index()]++;

  // Initialize the number of connections
  connectivity.init(tmp);

  // Reset current position for each entity
  std::fill(tmp.begin(), tmp.end(), 0);

  // Add the connections
  for (MeshEntityIterator e1(mesh, d1); !e1.end(); ++e1)
    for (MeshEntityIterator e0(*e1, d0); !e0.end(); ++e0)
      connectivity.set(e0->index(), e1->index(), tmp[e0->index()]++);
}
//----------------------------------------------------------------------------
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
  std::vector<std::vector<std::size_t> > connectivity(topology.size(d0));

  // A bitmap used to ensure we do not store duplicates
  std::vector<bool> e1_visited(topology.size(d1));

  // Iterate over all entities of dimension d0
  std::size_t max_size = 1;
  const std::size_t e0_num_entities = mesh.type().num_vertices(d0);
  const std::size_t e1_num_entities = mesh.type().num_vertices(d1);
  std::vector<std::size_t> _e0(e0_num_entities);
  std::vector<std::size_t> _e1(e1_num_entities);
  for (MeshEntityIterator e0(mesh, d0); !e0.end(); ++e0)
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
