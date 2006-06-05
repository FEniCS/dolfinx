// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-02
// Last changed: 2006-06-05

#include <dolfin/dolfin_log.h>
#include <dolfin/NewMesh.h>
#include <dolfin/MeshTopology.h>
#include <dolfin/MeshConnectivity.h>
#include <dolfin/MeshEntityIterator.h>
#include <dolfin/MeshAlgorithms.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void MeshAlgorithms::computeEntities(NewMesh& mesh, uint dim)
{
  dolfin_info("Computing mesh entities of topological dimension %d.", dim);
  
}
//-----------------------------------------------------------------------------
void MeshAlgorithms::computeConnectivity(NewMesh& mesh, uint d0, uint d1)
{
  dolfin_info("Computing mesh connectivity %d - %d.", d0, d1);

  // Get mesh topology and connectivity
  MeshTopology& topology = mesh.data.topology;
  MeshConnectivity& connectivity = topology(d0, d1);

  // Check if connectivity has already been computed
  if ( connectivity.size() > 0 )
  {
    cout << "Connectivity already computed, don't need to recompute." << endl;
    return;
  }

  // Decide how to compute the connectivity
  if ( d0 < d1 )
  {
    // Compute connectivity d1 - d0 and take transpose
    computeConnectivity(mesh, d1, d0);
    computeTranspose(mesh, d1, d0);
  }
  else if ( d0 == d1 )
  {

    // Compute connectivity d - 0 and 0 - d and use to compute diagonal
    computeConnectivity(mesh, d0, 0);
    computeConnectivity(mesh, 0, d0);
    computeDiagonal(mesh, d0);
  }
  else
  {
    // Compute connectivity dim - dim and generate d0 - d1
    computeConnectivity(mesh, topology.dim(), topology.dim());
    generateConnectivity(mesh, d0, d1);
  }
}
//----------------------------------------------------------------------------
void MeshAlgorithms::computeTranspose(NewMesh& mesh, uint d0, uint d1)
{
  // The transpose is computed in three steps:
  //
  //   1. Iterate over entities of dimension d0 and count the number
  //      of connections for each entity of dimension d1
  //
  //   2. Allocate memory / prepare data structures
  //
  //   3. Iterate again over entities of dimension d0 and add connections
  //      for each entity of dimension d1

  dolfin_info("Computing transpose of mesh connectivity %d - %d.", d0, d1);
  
  // Get mesh topology and connectivity
  MeshTopology& topology = mesh.data.topology;
  MeshConnectivity& connectivity = topology(d1, d0);

  // Need connectivity d0 - d1
  dolfin_assert(topology(d0, d1).size() > 0);

  // Temporary array
  Array<uint> tmp(topology.size(d1));

  // Reset size for each entity
  for (uint i = 0; i < tmp.size(); i++)
    tmp[i] = 0;

  // Count the number of connections
  for (MeshEntityIterator e0(mesh, d0); !e0.end(); ++e0)
    for (MeshEntityIterator e1(e0, d1); !e1.end(); ++e1)
      tmp[e1->index()]++;

  for (uint i = 0; i < tmp.size(); i++)
    cout << tmp[i] << endl;

  // Initialize the number of connections
  connectivity.init(tmp);

  connectivity.disp();

  // Reset current position for each entity
  for (uint i = 0; i < tmp.size(); i++)
    tmp[i] = 0;
  
  // Add the connections
  for (MeshEntityIterator e0(mesh, d0); !e0.end(); ++e0)
    for (MeshEntityIterator e1(e0, d1); !e1.end(); ++e1)
      connectivity.set(e1->index(), e0->index(), tmp[e1->index()]++);

  topology.disp();
}
//----------------------------------------------------------------------------
void MeshAlgorithms::computeDiagonal(NewMesh& mesh, uint d)
{
  dolfin_info("Computing diagonal mesh connectivity %d - %d.", d, d);

  // Get mesh topology and connectivity
  MeshTopology& topology = mesh.data.topology;
  //MeshConnectivity& connectivity = topology(d, d);
  uint dim = topology.dim();

  // Need connectivity d - 0 and 0 - d
  dolfin_assert(topology(dim, 0).size() > 0);
  dolfin_assert(topology(0, dim).size() > 0);

  // Entities connected if they share a common vertex
  uint d1 = 0;

  // Special case: vertices connected if they share an edge
  if ( d == 0 )
    d1 = 1;

  // Iterate over all entities of dimension d
  for (MeshEntityIterator e0(mesh, d); !e0.end(); ++e0)
  {
    // Iterate over all connected entities
    for (MeshEntityIterator e1(e0); !e1.end(); ++e1)
    {


    }
  }
}
//----------------------------------------------------------------------------
void MeshAlgorithms::generateConnectivity(NewMesh& mesh, uint d0, uint d1)
{
  dolfin_info("Generating mesh connectivity %d - %d from cell - vertex connectivity.", d0, d1);

  // Get mesh topology and connectivity
  MeshTopology& topology = mesh.data.topology;
  //MeshConnectivity& connectivity = topology(d0, d1);
  uint dim = topology.dim();

  // Need d0 > d1
  dolfin_assert(d0 > d1);

  // Need connectivity dim - 0
  dolfin_assert(topology(dim, 0).size() > 0);

  // Need connectivity dim - dim
  dolfin_assert(topology(dim, dim).size() > 0);

  // Get cell type and size of cell
  //CellType* cell_type = mesh.data.cell_type;
  // dolfin_assert(cell_type);
  //uint num_entities = cell_type->size(d1);
  // cout << "num_entities = " << num_entities << endl;

  // Iterate over all cells
  for (MeshEntityIterator c(mesh, topology.dim()); !c.end(); ++c)
  {
    // Get vertices
    //const uint num_vertices = cv.size(c->index());
    //const uint* vertices = cv(c->index());

    //cout << num_vertices << " " << vertices[0] << endl;
  }
}
//----------------------------------------------------------------------------
