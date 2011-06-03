// Copyright (C) 2006-2010 Anders Logg
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
// First added:  2006-06-02
// Last changed: 2011-03-17

#include <set>
#include <vector>
#include <boost/unordered_set.hpp>

#include <dolfin/common/Timer.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/utils.h>
#include <dolfin/log/dolfin_log.h>
#include "CellType.h"
#include "Mesh.h"
#include "MeshConnectivity.h"
#include "MeshEntity.h"
#include "MeshEntityIterator.h"
#include "MeshTopology.h"
#include "TopologyComputation.h"

using namespace dolfin;

// Set typedefs
typedef std::set<dolfin::uint> set;
typedef std::set<dolfin::uint>::const_iterator set_iterator;
//typedef boost::unordered_set<dolfin::uint> set;
//typedef boost::unordered_set<dolfin::uint>::const_iterator set_iterator;

//-----------------------------------------------------------------------------
dolfin::uint TopologyComputation::compute_entities(Mesh& mesh, uint dim)
{
  // Generating an entity of topological dimension dim is equivalent
  // to generating the connectivity dim - 0 (connections to vertices)
  // and the connectivity mesh.topology().dim() - dim (connections from cells).
  //
  // We generate entities by iterating over all cells and generating a
  // new entity only on its first occurence. Entities also contained
  // in a previously visited cell are not generated. The new entities
  // are computed in three steps:
  //
  //   1. Iterate over cells and count new entities
  //
  //   2. Allocate memory / prepare data structures
  //
  //   3. Iterate over cells and add new entities

  // Get mesh topology and connectivity
  MeshTopology& topology = mesh.topology();
  MeshConnectivity& ce = topology(topology.dim(), dim);
  MeshConnectivity& ev = topology(dim, 0);

  // Check if entities have already been computed
  if (topology.size(dim) > 0)
  {
    // Make sure we really have the connectivity
    if ((ce.size() == 0 && dim != topology.dim()) || (ev.size() == 0 && dim != 0))
      error("Entities of topological dimension %d exist but connectivity is missing.", dim);
    return topology.size(dim);
  }

  // Make sure connectivity does not already exist
  if (ce.size() > 0 || ev.size() > 0)
    error("Connectivity for topological dimension %d exists but entities are missing.", dim);

  //info("Computing mesh entities of topological dimension %d.", dim);

  // Compute connectivity dim - dim if not already computed
  compute_connectivity(mesh, mesh.topology().dim(), mesh.topology().dim());

  // Start timer
  //info("Creating mesh entities of dimension %d.", dim);
  Timer timer("compute entities dim = " + to_string(dim));

  // Get cell type
  const CellType& cell_type = mesh.type();

  // Initialize local array of entities
  const uint m = cell_type.num_entities(dim);
  const uint n = cell_type.num_vertices(dim);
  uint** entities = new uint*[m];
  for (uint i = 0; i < m; i++)
  {
    entities[i] = new uint[n];
    for (uint j = 0; j < n; j++)
      entities[i][j] = 0;
  }

  // Count the number of entities
  uint num_entities = 0;
  for (MeshEntityIterator c(mesh, mesh.topology().dim()); !c.end(); ++c)
  {
    // Get vertices from cell
    const uint* vertices = c->entities(0);
    assert(vertices);

    // Create entities
    cell_type.create_entities(entities, dim, vertices);

    // Count new entities
    num_entities += count_entities(mesh, *c, entities, m, n, dim);
  }

  // Initialize the number of entities and connections
  topology.init(dim, num_entities);
  ce.init(mesh.num_cells(), m);
  ev.init(num_entities, n);

  // Add new entities
  uint current_entity = 0;
  for (MeshEntityIterator c(mesh, mesh.topology().dim()); !c.end(); ++c)
  {
    // Get vertices from cell
    const uint* vertices = c->entities(0);
    assert(vertices);

    // Create entities
    cell_type.create_entities(entities, dim, vertices);

    // Add new entities to the mesh
    add_entities(mesh, *c, entities, m, n, dim, ce, ev, current_entity);
  }

  // Delete temporary data
  for (uint i = 0; i < m; i++)
    delete [] entities[i];
  delete [] entities;

  //info("Created %d new entities.", num_entities);

  return num_entities;
}
//-----------------------------------------------------------------------------
void TopologyComputation::compute_connectivity(Mesh& mesh, uint d0, uint d1)
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
  if (connectivity.size() > 0)
    return;

  //info("Computing mesh connectivity %d - %d.", d0, d1);

  // Compute entities if they don't exist
  if (topology.size(d0) == 0)
    compute_entities(mesh, d0);
  if (topology.size(d1) == 0)
    compute_entities(mesh, d1);

  // Check is mesh has entities
  if (topology.size(d0) == 0 && topology.size(d1) == 0)
    return;

  // Check if connectivity still needs to be computed
  if (connectivity.size() > 0)
    return;

  // Start timer
  //info("Computing mesh connectivity %d - %d.", d0, d1);
  Timer timer("compute connectivity " + to_string(d0) + " - " + to_string(d1));

  // Decide how to compute the connectivity
  if (d0 < d1)
  {
    // Compute connectivity d1 - d0 and take transpose
    compute_connectivity(mesh, d1, d0);
    compute_from_transpose(mesh, d0, d1);
  }
  else
  {
    // These connections should already exist
    assert(!(d0 > 0 && d1 == 0));

    // Choose how to take intersection
    uint d = 0;
    if (d0 == 0 && d1 == 0)
      d = mesh.topology().dim();

    // Compute connectivity d0 - d - d1 and take intersection
    compute_connectivity(mesh, d0, d);
    compute_connectivity(mesh, d, d1);
    compute_from_intersection(mesh, d0, d1, d);
  }
}
//----------------------------------------------------------------------------
void TopologyComputation::compute_from_transpose(Mesh& mesh, uint d0, uint d1)
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
  assert(topology(d1, d0).size() > 0);

  // Temporary array
  std::vector<uint> tmp(topology.size(d0));

  // Reset size for each entity
  for (uint i = 0; i < tmp.size(); i++)
    tmp[i] = 0;

  // Count the number of connections
  for (MeshEntityIterator e1(mesh, d1); !e1.end(); ++e1)
    for (MeshEntityIterator e0(*e1, d0); !e0.end(); ++e0)
      tmp[e0->index()]++;

  // Initialize the number of connections
  connectivity.init(tmp);

  // Reset current position for each entity
  for (uint i = 0; i < tmp.size(); i++)
    tmp[i] = 0;

  // Add the connections
  for (MeshEntityIterator e1(mesh, d1); !e1.end(); ++e1)
    for (MeshEntityIterator e0(*e1, d0); !e0.end(); ++e0)
      connectivity.set(e0->index(), e1->index(), tmp[e0->index()]++);
}
//----------------------------------------------------------------------------
void TopologyComputation::compute_from_intersection(Mesh& mesh,
                                                    uint d0, uint d1, uint d)
{
  log(TRACE, "Computing mesh connectivity %d - %d from intersection %d - %d - %d.",
      d0, d1, d0, d, d1);

  // Get mesh topology
  MeshTopology& topology = mesh.topology();

  // Check preconditions
  assert(d0 >= d1);
  assert(topology(d0, d).size() > 0);
  assert(topology(d, d1).size() > 0);

  // Temporary dynamic storage, later copied into static storage
  std::vector<std::vector<uint> > connectivity(topology.size(d0));

  // Iterate over all entities of dimension d0
  uint max_size = 1;
  for (MeshEntityIterator e0(mesh, d0); !e0.end(); ++e0)
  {
    // Get set of connected entities for current entity
    std::vector<uint>& entities = connectivity[e0->index()];

    // Reserve space
    entities.reserve(max_size);

    // Iterate over all connected entities of dimension d
    for (MeshEntityIterator e(*e0, d); !e.end(); ++e)
    {
      // Iterate over all connected entities of dimension d1
      for (MeshEntityIterator e1(*e, d1); !e1.end(); ++e1)
      {
        if (d0 == d1)
        {
          // An entity is not a neighbor to itself
          if (e0->index() != e1->index() && std::find(entities.begin(), entities.end(), e1->index()) == entities.end())
            entities.push_back(e1->index());
        }
        else
        {
          // Entity e1 must be completely contained in e0
          if (contains(*e0, *e1) && std::find(entities.begin(), entities.end(), e1->index()) == entities.end())
            entities.push_back(e1->index());
        }
      }
    }

    // Store maximum size
    if (entities.size() > max_size)
      max_size = entities.size();
  }

  // Copy to static storage
  topology(d0, d1).set(connectivity);
}
//-----------------------------------------------------------------------------
dolfin::uint TopologyComputation::count_entities(Mesh& mesh, MeshEntity& cell,
                                                 uint** entities, uint m, uint n,
                                                 uint dim)
{
  // For each entity, we iterate over connected and previously visited
  // cells to see if the entity has already been counted.

  // Needs to be a cell
  assert(cell.dim() == mesh.topology().dim());

  // Iterate over the given list of entities
  uint num_entities = 0;
  for (uint i = 0; i < m; i++)
  {
    // Iterate over connected cells and look for entity
    for (MeshEntityIterator c(cell, mesh.topology().dim()); !c.end(); ++c)
    {
      // Check only previously visited cells
      if (c->index() >= cell.index())
        continue;

      // Check for vertices
      if (contains(c->entities(0), c->num_entities(0), entities[i], n))
        goto found;
    }

    // Increase counter
    num_entities++;

    // Entity found, don't need to count
    found:
    ;
  }

  return num_entities;
}
//----------------------------------------------------------------------------
void TopologyComputation::add_entities(Mesh& mesh, MeshEntity& cell,
				 uint** entities, uint m, uint n, uint dim,
				 MeshConnectivity& ce,
				 MeshConnectivity& ev,
				 uint& current_entity)
{
  // We repeat the same algorithm as in count_entities() but this time
  // we add any entities that are new.

  // Needs to be a cell
  assert(cell.dim() == mesh.topology().dim());

  // Iterate over the given list of entities
  for (uint i = 0; i < m; i++)
  {
    // Iterate over connected cells and look for entity
    for (MeshEntityIterator c(cell, mesh.topology().dim()); !c.end(); ++c)
    {
      // Check only previously visited cells
      if (c->index() >= cell.index())
        continue;

      // Check all entities of dimension dim in connected cell
      uint num_other_entities = c->num_entities(dim);
      const uint* other_entities = c->entities(dim);
      for (uint j = 0; j < num_other_entities; j++)
      {
        // Can't use iterators since connectivity has not been computed
        MeshEntity e(mesh, dim, other_entities[j]);
        if (contains(e.entities(0), e.num_entities(0), entities[i], n))
        {
          // Entity already exists, so pick the index
          ce.set(cell.index(), e.index(), i);
          goto found;
        }
      }
    }

    // Entity does not exist, so create it
    ce.set(cell.index(), current_entity, i);
    ev.set(current_entity, entities[i]);

    // Increase counter
    current_entity++;

    // Entity found, don't need to create
    found:
    ;
  }
}
//----------------------------------------------------------------------------
bool TopologyComputation::contains(MeshEntity& e0, MeshEntity& e1)
{
  // Check vertices
  return contains(e0.entities(0), e0.num_entities(0),
		  e1.entities(0), e1.num_entities(0));
}
//----------------------------------------------------------------------------
bool TopologyComputation::contains(const uint* v0, uint n0, const uint* v1, uint n1)
{
  assert(v0);
  assert(v1);

  for (uint i1 = 0; i1 < n1; i1++)
  {
    bool found = false;
    for (uint i0 = 0; i0 < n0; i0++)
    {
      if (v0[i0] == v1[i1])
      {
        found = true;
        break;
      }
    }
    if (!found)
      return false;
  }

  return true;
}
//----------------------------------------------------------------------------
