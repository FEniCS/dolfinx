// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-09
// Last changed: 2006-05-22

#include <dolfin/MeshConnectivity.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshConnectivity::MeshConnectivity()
  : _size(0), num_entities(0), connections(0), offsets(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshConnectivity::~MeshConnectivity()
{
  clear();
}
//-----------------------------------------------------------------------------
void MeshConnectivity::clear()
{
  _size = 0;
  num_entities = 0;

  if ( connections )
    delete [] connections;
  connections = 0;

  if ( offsets )
    delete [] offsets;
  offsets = 0;
}
//-----------------------------------------------------------------------------
void MeshConnectivity::init(uint num_entities, uint num_connections)
{
  // Clear old data if any
  clear();
  
  // Compute the total size
  _size = num_entities*num_connections;
  this->num_entities = num_entities;

  // Allocate data
  connections = new uint[_size];
  offsets = new uint[num_entities + 1];
  
  // Initialize data
  for (uint i = 0; i < _size; i++)
    connections[i] = 0;
  for (uint e = 0; e <= num_entities; e++)
    offsets[e] = e*num_connections;
}
//-----------------------------------------------------------------------------
void MeshConnectivity::init(uint num_entities, Array<uint>& num_connections)
{
  dolfin_assert(num_entities == num_connections.size());

  // Clear old data if any
  clear();

  // Compute the total size  
  _size = 0;
  this->num_entities = num_entities;
  for (uint e = 0; e < num_entities; e++)
    _size += num_connections[e];
  
  // Allocate data
  connections = new uint[_size];
  offsets = new uint[num_entities + 1];
  
  // Initialize data
  uint offset = 0;
  for (uint e = 0; e < num_entities; e++)
  {
    offsets[e] = offset;
    for (uint i = 0; i < num_connections[i]; i++)
      connections[offset++] = 0;
  }
  offsets[num_entities] = offset;
}
//-----------------------------------------------------------------------------
void MeshConnectivity::set(uint entity, Array<uint>& connections)
{
  dolfin_assert(entity < num_entities);
  dolfin_assert(connections.size() == offsets[entity + 1] - offsets[entity]);
  
  // Copy data
  for (uint i = 0; i < connections.size(); i++)
    this->connections[offsets[entity] + i] = connections[i];
}
//-----------------------------------------------------------------------------
void MeshConnectivity::set(Array<Array<uint> >& connections)
{
  // Clear old data if any
  clear();

  // Compute the total size
  _size = 0;
  num_entities = connections.size();
  for (uint e = 0; e < num_entities; e++)
    _size += connections[e].size();

  // Allocate data
  this->connections = new uint[_size];
  offsets = new uint[num_entities + 1];

  // Copy data
  uint offset = 0;
  for (uint e = 0; e < num_entities; e++)
  {
    offsets[e] = offset;
    for (uint c = 0; c < connections[e].size(); c++)
      this->connections[offset++] = connections[e][c];
  }
  offsets[num_entities] = offset;
}
//-----------------------------------------------------------------------------
void MeshConnectivity::disp() const
{
  // Check if there are any connections
  if ( _size == 0 )
  {
    cout << "empty" << endl;
    return;
  }

  // Display all connections
  for (uint e = 0; e < num_entities; e++)
  {
    cout << e << ":";
    for (uint i = offsets[e]; i < offsets[e + 1]; i++)
      cout << " " << connections[i];
    cout << endl;
  }
}
//-----------------------------------------------------------------------------
