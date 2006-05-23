// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-08
// Last changed: 2006-05-23

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
MeshTopology::~MeshTopology()
{
  clear();
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
  connectivity = new MeshConnectivity* [dim + 1];
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
void MeshTopology::set(uint d0, uint d1, Array< Array<uint> >& connectivity)
{
  dolfin_assert(d0 <= _dim);
  dolfin_assert(d1 <= _dim);

  this->connectivity[d0][d1].set(connectivity);
}
//-----------------------------------------------------------------------------
void MeshTopology::disp() const
{
  cout << "Mesh topology" << endl;
  cout << "-------------" << endl << endl;

  // Begin indentation
  dolfin_begin();
  
  // Display topological dimension
  cout << "Topological dimension: " << _dim << endl << endl;
  
  // Display number of entities for each topological dimension
  cout << "Number of entities" << endl << endl;
  dolfin_begin();
  for (uint d = 0; d <= _dim; d++)
    cout << "dim = " << d << ": " << num_entities[d] << endl;
  dolfin_end();
  cout << endl;
  
  // Display connectivity for each topological dimension
  for (uint d0 = 0; d0 <= _dim; d0++)
  {
    for (uint d1 = 0; d1 <= _dim; d1++)
    {
      cout << "Connectivity: " << d0 << " -- " << d1 << endl << endl;
      dolfin_begin();
      connectivity[d0][d1].disp();
      dolfin_end();
      cout << endl;
    }
  }

  // End indentation
  dolfin_end();
}
//-----------------------------------------------------------------------------
