// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-02
// Last changed: 2006-06-06

#include <set>
#include <dolfin/dolfin_log.h>
#include <dolfin/CellType.h>
#include <dolfin/NewMesh.h>
#include <dolfin/MeshTopology.h>
#include <dolfin/MeshConnectivity.h>
#include <dolfin/MeshEntityIterator.h>
#include <dolfin/MeshAlgorithms.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void MeshAlgorithms::computeEntities(NewMesh& mesh, uint dim)
{
  // Generating an entity of topological dimension dim is equivalent
  // to generating the connectivity dim - 0 (connections to vertices).
  
  // Get mesh topology and connectivity
  MeshTopology& topology = mesh.data.topology;
  MeshConnectivity& connectivity = topology(dim, 0);
  
  // Check if entities have already been computed
  if ( topology.size(dim) > 0 )
  {
    // Make sure we really have the connectivity
    if ( connectivity.size() == 0 && dim != 0 )
      dolfin_error1("Entities of topological dimension %d exist but connectivity is missing.", dim);
    return;
  }

  dolfin_info("Generating mesh entities of topological dimension %d.", dim);

  // Compute connectivity dim - dim needed to generate entities
  computeConnectivity(mesh, mesh.dim(), mesh.dim());

  // Count the number of entities
  uint num_entities = countEntities(mesh, dim);

  cout << "Number of entities: " << num_entities << endl;

  // Add entities
  addEntities(mesh, dim, num_entities);
}
//-----------------------------------------------------------------------------
void MeshAlgorithms::computeConnectivity(NewMesh& mesh, uint d0, uint d1)
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
  MeshTopology& topology = mesh.data.topology;
  MeshConnectivity& connectivity = topology(d0, d1);

  // Check if connectivity has already been computed
  if ( connectivity.size() > 0 )
    return;

  dolfin_info("Computing mesh connectivity %d - %d.", d0, d1);
  dolfin_begin();

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
      d = mesh.dim();

    // Compute connectivity d0 - d - d1 and take intersection
    computeConnectivity(mesh, d0, d);
    computeConnectivity(mesh, d, d1);
    computeFromIntersection(mesh, d0, d1, d);
  }

  dolfin_end();
}
//----------------------------------------------------------------------------
void MeshAlgorithms::computeFromTranspose(NewMesh& mesh, uint d0, uint d1)
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
  MeshTopology& topology = mesh.data.topology;
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

  cout << "Transpose computed" << endl;
}
//----------------------------------------------------------------------------
void MeshAlgorithms::computeFromIntersection(NewMesh& mesh,
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
  MeshTopology& topology = mesh.data.topology;
  MeshConnectivity& connectivity = topology(d0, d1);

  // Need connectivity d0 - d and d - d1
  dolfin_assert(topology(d0, d).size() > 0);
  dolfin_assert(topology(d, d1).size() > 0);

  // Temporary array
  Array<uint> tmp(topology.size(d0));

  // Reset size for each entity
  for (uint i = 0; i < tmp.size(); i++)
    tmp[i] = 0;

  // FIXME: Check how efficient this is. Maybe a vector is better.
  // FIXME: What happens at clear()? Do we allocate new memory all the time?

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
	if ( e0->index() != e1->index() )
	  entities.insert(e1->index());
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

    // Iterate over all connected entities of dimension d1
    for (MeshEntityIterator e(e0, d); !e.end(); ++e)
    {
      // Iterate over all connected entities of dimension d
      for (MeshEntityIterator e1(e, d1); !e1.end(); ++e1)
      {
	if ( e0->index() != e1->index() )
	  entities.insert(e1->index());
      }
    }

    // Add the connected entities
    uint pos = 0;
    for (std::set<uint>::iterator it = entities.begin(); it != entities.end(); ++it)
      connectivity.set(e0->index(), *it, pos++);
  }
}
//----------------------------------------------------------------------------
dolfin::uint MeshAlgorithms::findEntities(NewMesh& mesh, uint dim,
					  bool only_count)
{
  // We generate entities by iterating over all cells and generating a
  // new entity only on its first occurence. Entities also contained
  // in a previously visited cell are not generated. If only_count is
  // set, then we just count the total number of entities.

  // Get mesh topology and connectivity
  MeshTopology& topology = mesh.data.topology;
  MeshConnectivity& connectivity = topology(dim, d1);

  // Need connectivity dim - dim
  dolfin_assert(topology(mesh.dim(), mesh.dim()).size() > 0);

  // Need data to be initialized when not counting
  dolfin_assert(only_count || topology.size(dim) > 0);

  // Get cell type
  CellType* cell_type = mesh.data.cell_type;
  dolfin_assert(cell_type);

  // Initialize local array of entities
  const uint m = cell_type->numEntities(dim);
  const uint n = cell_type->numVertices(dim);
  Array<Array<uint> > entities(m);
  for (uint i = 0; i < m; i++)
    for (uint j = 0; j < n; j++)
      entities[i].push_back(0);

  // Reset the number of entities
  uint num_entities = 0;

  // Iterate over all cells and count entities of given dimension
  for (MeshEntityIterator c(mesh, mesh.dim()); !c.end(); ++c)
  {
    // Get vertices from cell
    const uint* vertices = c->connections(0);
    dolfin_assert(vertices);

    // Create entities
    cell_type->createEntities(entities, dim, vertices);

    // Count only entities which have not previously been counted
    for (uint i = 0; i < entities.size(); i++)
    {
      bool found = false;
      for (MeshEntityIterator c1(c, mesh.dim()); !c1.end(); ++c1)
      {
	// Check only previously visited and connected cells
	if ( c1->index() >= c->index() )
	continue;
      
	// Check if entity contains all vertices in entity
	if ( containsVertices(*c1, entities[i]) )
	{
	  found = true;
	  break;
	}
      }

      // Skip entities contained in previously visited cells
      if ( found )
	continue;

      // Count or create entity
      if ( only_count )
	num_entities++;
      else
	cout << "Creating entity" << endl;
    }
  }

  return num_entities;
}
//----------------------------------------------------------------------------
void MeshAlgorithms::addEntities(NewMesh& mesh, uint dim, uint num_entities)
{
  // We repeat the same algorithm as in countEntitites() but count the number of entities by iterating over all cells and
  // counting new entities only on their first occurence. Entities
  // also contained in a previously visited cell are not counted.

  // Need connectivity dim - dim
  dolfin_assert(mesh.data.topology(mesh.dim(), mesh.dim()).size() > 0);

  // Get cell type
  CellType* cell_type = mesh.data.cell_type;
  dolfin_assert(cell_type);

  // Initialize local array of entities
  const uint m = cell_type->numEntities(dim);
  const uint n = cell_type->numVertices(dim);
  Array<Array<uint> > entities(m);
  for (uint i = 0; i < m; i++)
    for (uint j = 0; j < n; j++)
      entities[i].push_back(0);

  // Reset the number of entities
  uint num_entities = 0;

  // Iterate over all cells and count entities of given dimension
  for (MeshEntityIterator c(mesh, mesh.dim()); !c.end(); ++c)
  {
    // Get vertices from cell
    const uint* vertices = c->connections(0);
    dolfin_assert(vertices);

    // Create entities
    cell_type->createEntities(entities, dim, vertices);

    // Count only entities which have not previously been counted
    for (uint i = 0; i < entities.size(); i++)
    {
      bool found = false;
      for (MeshEntityIterator c1(c, mesh.dim()); !c1.end(); ++c1)
      {
	// Check only previously visited and connected cells
	if ( c1->index() >= c->index() )
	continue;
      
	// Check if entity contains all vertices in entity
	if ( containsVertices(*c1, entities[i]) )
	{
	  found = true;
	  break;
	}
      }

      // Did not find entity so count it
      if ( !found )
	num_entities++;
    }
  }

  return num_entities;
}
//----------------------------------------------------------------------------
bool MeshAlgorithms::containsVertices(MeshEntity& entity, Array<uint>& vertices)
{
  // Iterate over all vertices and check if it is contained
  for (uint i = 0; i < vertices.size(); i++)
  {
    bool found = false;
    for (MeshEntityIterator v(entity, 0); !v.end(); ++v)
    {
      if ( vertices[i] == v->index() )
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
