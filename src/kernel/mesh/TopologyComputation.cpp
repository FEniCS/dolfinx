// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-02
// Last changed: 2006-10-11

#include <set>
#include <dolfin/dolfin_log.h>
#include <dolfin/Array.h>
#include <dolfin/CellType.h>
#include <dolfin/NewMesh.h>
#include <dolfin/MeshTopology.h>
#include <dolfin/MeshConnectivity.h>
#include <dolfin/MeshEntity.h>
#include <dolfin/MeshEntityIterator.h>
#include <dolfin/TopologyComputation.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::uint TopologyComputation::computeEntities(NewMesh& mesh, uint dim)
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
  if ( topology.size(dim) > 0 )
  {
    // Make sure we really have the connectivity
    if ( ce.size() == 0 || (ev.size() == 0 && dim != 0) )
      dolfin_error1("Entities of topological dimension %d exist but connectivity is missing.", dim);
    return topology.size(dim);
  }
  else
  {
    // Make sure connectivity does not already exist
    if ( ce.size() > 0 || ev.size() > 0 )
      dolfin_error1("Connectivity for topological dimension %d exists but entities are missing.", dim);
  }

  dolfin_info("Computing mesh entities of topological dimension %d.", dim);

  // Compute connectivity dim - dim if not already computed
  computeConnectivity(mesh, mesh.topology().dim(), mesh.topology().dim());

  // Get cell type
  const CellType& cell_type = mesh.type();

  // Initialize local array of entities
  const uint m = cell_type.numEntities(dim);
  const uint n = cell_type.numVertices(dim);
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
    const uint* vertices = c->connections(0);
    dolfin_assert(vertices);
    
    // Create entities
    cell_type.createEntities(entities, dim, vertices);
    
    // Count new entities
    num_entities += countEntities(mesh, *c, entities, m, n, dim);
  }

  // Initialize the number of entities and connections
  topology.init(dim, num_entities);
  ce.init(mesh.numCells(), m);
  ev.init(num_entities, n);

  // Add new entities
  uint current_entity = 0;
  for (MeshEntityIterator c(mesh, mesh.topology().dim()); !c.end(); ++c)
  {
    // Get vertices from cell
    const uint* vertices = c->connections(0);
    dolfin_assert(vertices);
    
    // Create entities
    cell_type.createEntities(entities, dim, vertices);
    
    // Add new entities to the mesh
    addEntities(mesh, *c, entities, m, n, dim, ce, ev, current_entity);
  }

  // Delete temporary data
  for (uint i = 0; i < m; i++)
    delete [] entities[i];
  delete [] entities;

  dolfin_info("Created %d new entities.", num_entities);

  return num_entities;
}
//-----------------------------------------------------------------------------
void TopologyComputation::computeConnectivity(NewMesh& mesh, uint d0, uint d1)
{
  // This is where all the logic takes place to find a stragety for
  // the connectivity computation. For any given pair (d0, d1), the
  // connectivity is computed by suitably combining the following
  // basic building blocks:
  //
  //   1. computeEntities():     d  - 0  from dim - 0
  //   2. computeTranspose():    d0 - d1 from d1 - d0
  //   3. computeIntersection(): d0 - d1 from d0 - d' - d1
  //
  // Each of these functions assume a set of preconditions that we
  // need to satisfy.

  // Get mesh topology and connectivity
  MeshTopology& topology = mesh.topology();
  MeshConnectivity& connectivity = topology(d0, d1);

  // Check if connectivity has already been computed
  if ( connectivity.size() > 0 )
    return;

  dolfin_info("Computing mesh connectivity %d - %d.", d0, d1);

  // Compute entities if they don't exist
  if ( topology.size(d0) == 0 )
    computeEntities(mesh, d0);
  if ( topology.size(d1) == 0 )
    computeEntities(mesh, d1);

  // Check if connectivity still needs to be computed
  if ( connectivity.size() > 0 )
    return;

  // Decide how to compute the connectivity
  if ( d0 < d1 )
  {
    // Compute connectivity d1 - d0 and take transpose
    computeConnectivity(mesh, d1, d0);
    computeFromTranspose(mesh, d0, d1);
  }
  else
  {
    // These connections should already exist
    dolfin_assert(!(d0 > 0 && d1 == 0));

    // Choose how to take intersection
    uint d = 0;
    if ( d0 == 0 && d1 == 0 )
      d = mesh.topology().dim();

    // Compute connectivity d0 - d - d1 and take intersection
    computeConnectivity(mesh, d0, d);
    computeConnectivity(mesh, d, d1);
    computeFromIntersection(mesh, d0, d1, d);
  }
}
//----------------------------------------------------------------------------
void TopologyComputation::computeFromTranspose(NewMesh& mesh, uint d0, uint d1)
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

  dolfin_info("Computing mesh connectivity %d - %d from transpose.", d0, d1);
  
  // Get mesh topology and connectivity
  MeshTopology& topology = mesh.topology();
  MeshConnectivity& connectivity = topology(d0, d1);

  // Need connectivity d1 - d0
  dolfin_assert(topology(d1, d0).size() > 0);

  // Temporary array
  Array<uint> tmp(topology.size(d0));

  // Reset size for each entity
  for (uint i = 0; i < tmp.size(); i++)
    tmp[i] = 0;

  // Count the number of connections
  for (MeshEntityIterator e1(mesh, d1); !e1.end(); ++e1)
    for (MeshEntityIterator e0(e1, d0); !e0.end(); ++e0)
      tmp[e0->index()]++;

  // Initialize the number of connections
  connectivity.init(tmp);

  // Reset current position for each entity
  for (uint i = 0; i < tmp.size(); i++)
    tmp[i] = 0;
  
  // Add the connections
  for (MeshEntityIterator e1(mesh, d1); !e1.end(); ++e1)
    for (MeshEntityIterator e0(e1, d0); !e0.end(); ++e0)
      connectivity.set(e0->index(), e1->index(), tmp[e0->index()]++);
}
//----------------------------------------------------------------------------
void TopologyComputation::computeFromIntersection(NewMesh& mesh,
					     uint d0, uint d1, uint d)
{
  // The intersection is computed in three steps:
  //
  //   1. Nested iteration over mesh - d0 - d - d1 and count the connections
  //
  //   2. Allocate memory / prepare data structures
  //
  //   3. Nested iteration over mesh - d0 - d - d1 and add the connections

  dolfin_info("Computing mesh connectivity %d - %d from intersection %d - %d - %d.",
	      d0, d1, d0, d, d1);

  // Get mesh topology and connectivity
  MeshTopology& topology = mesh.topology();
  MeshConnectivity& connectivity = topology(d0, d1);

  // Need d0 >= d1
  dolfin_assert(d0 >= d1);

  // Need connectivity d0 - d and d - d1
  dolfin_assert(topology(d0, d).size() > 0);
  dolfin_assert(topology(d, d1).size() > 0);

  // Temporary array
  Array<uint> tmp(topology.size(d0));

  // Reset size for each entity
  for (uint i = 0; i < tmp.size(); i++)
    tmp[i] = 0;

  // FIXME: Check efficiency of std::set, compare with std::vector

  // A set with connected entities
  std::set<uint> entities;

  // Iterate over all entities of dimension d0
  for (MeshEntityIterator e0(mesh, d0); !e0.end(); ++e0)
  {
    // Clear set of connected entities
    entities.clear();

    // Iterate over all connected entities of dimension d
    for (MeshEntityIterator e(e0, d); !e.end(); ++e)
    {
      // Iterate over all connected entities of dimension d1
      for (MeshEntityIterator e1(e, d1); !e1.end(); ++e1)
      {
	if ( d0 == d1 )
	{
	  // An entity is not a neighbor to itself
	  if ( e0->index() != e1->index() )
	    entities.insert(e1->index());
	}
	else
	{
	  // Entity e1 must be completely contained in e0
	  if ( contains(*e0, *e1) )
	    entities.insert(e1->index());
	}
      }
    }

    // Count the number of connected entities
    tmp[e0->index()] = entities.size();
  }

  // Initialize the number of connections
  connectivity.init(tmp);

  // Iterate over all entities of dimension d
  for (MeshEntityIterator e0(mesh, d0); !e0.end(); ++e0)
  {
    // Clear set of connected entities
    entities.clear();

    // Iterate over all connected entities of dimension d
    for (MeshEntityIterator e(e0, d); !e.end(); ++e)
    {
      // Iterate over all connected entities of dimension d1
      for (MeshEntityIterator e1(e, d1); !e1.end(); ++e1)
      {
	if ( d0 == d1 )
	{
	  // An entity is not a neighbor to itself
	  if ( e0->index() != e1->index() )
	    entities.insert(e1->index());
	}
	else
	{
	  // Entity e1 must be completely contained in e0
	  if ( contains(*e0, *e1) )
	    entities.insert(e1->index());
	}
      }
    }

    // Add the connected entities
    uint pos = 0;
    for (std::set<uint>::iterator it = entities.begin(); it != entities.end(); ++it)
      connectivity.set(e0->index(), *it, pos++);
  }
}
//----------------------------------------------------------------------------
dolfin::uint TopologyComputation::countEntities(NewMesh& mesh, MeshEntity& cell,
					   uint** entities, uint m, uint n,
					   uint dim)
{
  // For each entity, we iterate over connected and previously visited
  // cells to see if the entity has already been counted.
  
  // Needs to be a cell
  dolfin_assert(cell.dim() == mesh.topology().dim());

  // Iterate over the given list of entities
  uint num_entities = 0;
  for (uint i = 0; i < m; i++)
  {
    // Iterate over connected cells and look for entity
    for (MeshEntityIterator c(cell, mesh.topology().dim()); !c.end(); ++c)
    {
      // Check only previously visited cells
      if ( c->index() >= cell.index() )
	continue;

      // Check for vertices
      if ( contains(c->connections(0), c->numConnections(0), entities[i], n) )
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
void TopologyComputation::addEntities(NewMesh& mesh, MeshEntity& cell,
				 uint** entities, uint m, uint n, uint dim,
				 MeshConnectivity& ce,
				 MeshConnectivity& ev,
				 uint& current_entity)
{
  // We repeat the same algorithm as in countEntities() but this time
  // we add any entities that are new.
  
  // Needs to be a cell
  dolfin_assert(cell.dim() == mesh.topology().dim());

  // Iterate over the given list of entities
  for (uint i = 0; i < m; i++)
  {
    // Iterate over connected cells and look for entity
    for (MeshEntityIterator c(cell, mesh.topology().dim()); !c.end(); ++c)
    {
      // Check only previously visited cells
      if ( c->index() >= cell.index() )
	continue;
      
      // Check all entities of dimension dim in connected cell
      uint num_other_entities = c->numConnections(dim);
      uint* other_entities = c->connections(dim);
      for (uint j = 0; j < num_other_entities; j++)
      {
	// Can't use iterators since connectivity has not been computed
	MeshEntity e(mesh, dim, other_entities[j]);
	if ( contains(e.connections(0), e.numConnections(0), entities[i], n) )
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
  return contains(e0.connections(0), e0.numConnections(0),
		  e1.connections(0), e1.numConnections(0));
}
//----------------------------------------------------------------------------
bool TopologyComputation::contains(uint* v0, uint n0, uint* v1, uint n1)
{
  dolfin_assert(v0);
  dolfin_assert(v1);

  for (uint i1 = 0; i1 < n1; i1++)
  {
    bool found = false;
    for (uint i0 = 0; i0 < n0; i0++)
    {
      if ( v0[i0] == v1[i1] )
      {
	found = true;
	break;
      }
    }
    if ( !found )
      return false;
  }

  return true;
}
//----------------------------------------------------------------------------
