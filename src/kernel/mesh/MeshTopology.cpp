// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-05-08
// Last changed: 2006-11-01

#include <dolfin/dolfin_log.h>
#include <dolfin/MeshConnectivity.h>
#include <dolfin/MeshTopology.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshTopology::MeshTopology()
  : _dim(0), num_entities(0), connectivity(0)
  
{
  // Do nothing
}
//-----------------------------------------------------------------------------  
MeshTopology::MeshTopology(const MeshTopology& topology)
  : _dim(0), num_entities(0), connectivity(0)
{
  *this = topology;
}
//-----------------------------------------------------------------------------  
MeshTopology::~MeshTopology()
{
  clear();
}
//-----------------------------------------------------------------------------
const MeshTopology& MeshTopology::operator= (const MeshTopology& topology)
{
  // Clear old data if any
  clear();

  // Allocate data
  _dim = topology._dim;
  num_entities = new uint[_dim + 1];
  connectivity = new MeshConnectivity*[_dim + 1];
  for (uint d = 0; d <= _dim; d++)
    connectivity[d] = new MeshConnectivity[_dim + 1];

  // Copy data
  if ( _dim > 0 )
  {
    for (uint d = 0; d <= _dim; d++)
      num_entities[d] = topology.num_entities[d];
    for (uint d0 = 0; d0 <= _dim; d0++)
      for (uint d1 = 0; d1 <= _dim; d1++)
	connectivity[d0][d1] = topology.connectivity[d0][d1];
  }

  return *this;
}
//-----------------------------------------------------------------------------
void MeshTopology::clear()
{
  // Delete number of mesh entities
  if ( num_entities )
    delete [] num_entities;
  num_entities = 0;
  
  // Delete mesh connectivity
  if ( connectivity )
  {
    for (uint d = 0; d <= _dim; d++)
      delete [] connectivity[d];
    delete [] connectivity;
  }
  connectivity = 0;

  // Reset dimension
  _dim = 0;
}
//-----------------------------------------------------------------------------
void MeshTopology::init(uint dim)
{
  // Clear old data if any
  clear();

  // Initialize number of mesh entities
  num_entities = new uint[dim + 1];
  for (uint d = 0; d <= dim; d++)
    num_entities[d] = 0;

  // Initialize mesh connectivity
  connectivity = new MeshConnectivity*[dim + 1];
  for (uint d = 0; d <= dim; d++)
    connectivity[d] = new MeshConnectivity[dim + 1];

  // Save dimension
  _dim = dim;
}
//-----------------------------------------------------------------------------
void MeshTopology::init(uint dim, uint size)
{
  dolfin_assert(num_entities);
  dolfin_assert(dim <= _dim);

  num_entities[dim] = size;
}
//-----------------------------------------------------------------------------
void MeshTopology::disp() const
{
  cout << "Mesh topology" << endl;
  cout << "-------------" << endl << endl;

  // Begin indentation
  begin("");

  // Check if empty
  if ( _dim == 0 )
  {
    cout << "empty" << endl << endl;
    end();
    return;
  }
  
  // Display topological dimension
  cout << "Topological dimension: " << _dim << endl << endl;
  
  // Display number of entities for each topological dimension
  cout << "Number of entities:" << endl << endl;
  begin("");
  for (uint d = 0; d <= _dim; d++)
    cout << "dim = " << d << ": " << num_entities[d] << endl;
  end();
  cout << endl;
  
  // Display matrix of connectivities
  cout << "Connectivity:" << endl << endl;
  begin("");
  cout << " ";
  for (uint d1 = 0; d1 <= _dim; d1++)
    cout << " " << d1;
  cout << endl;
  for (uint d0 = 0; d0 <= _dim; d0++)
  {
    cout << d0;
    for (uint d1 = 0; d1 <= _dim; d1++)
    {
      if ( connectivity[d0][d1].size() > 0 )
	cout << " x";
      else
	cout << " -";
    }
    cout << endl;
  }
  cout << endl;
  end();

  // Display connectivity for each topological dimension
  for (uint d0 = 0; d0 <= _dim; d0++)
  {
    for (uint d1 = 0; d1 <= _dim; d1++)
    {
      if ( connectivity[d0][d1].size() == 0 )
	continue;
      cout << "Connectivity " << d0 << " -- " << d1 << ":" << endl << endl;
      begin("");
      connectivity[d0][d1].disp();
      end();
      cout << endl;
    }
  }

  // End indentation
  end();
}
//-----------------------------------------------------------------------------
