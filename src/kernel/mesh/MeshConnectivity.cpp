// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-09
// Last changed: 2006-06-05

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
void MeshConnectivity::init(Array<uint>& num_connections)
{
  // Clear old data if any
  clear();

  // Initialize offsets and compute total size
  num_entities = num_connections.size();
  offsets = new uint[num_entities + 1];
  _size = 0;
  for (uint e = 0; e < num_entities; e++)
  {
    offsets[e] = _size;
    _size += num_connections[e];
  }
  offsets[num_entities] = _size;
  
  // Initialize connections
  connections = new uint[_size];
  for (uint i = 0; i < _size; i++)
    connections[i] = 0;
}
//-----------------------------------------------------------------------------
void MeshConnectivity::set(uint entity, uint connection, uint pos)
{
  dolfin_assert(entity < num_entities);
  dolfin_assert(pos < offsets[entity + 1] - offsets[entity]);

  connections[offsets[entity] + pos] = connection;
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

  // Initialize offsets and compute total size
  num_entities = connections.size();
  offsets = new uint[num_entities + 1];
  _size = 0;
  for (uint e = 0; e < num_entities; e++)
  {
    offsets[e] = _size;
    _size += connections[e].size();
  }
  offsets[num_entities] = _size;
  
  // Initialize connections
  this->connections = new uint[_size];
  for (uint e = 0; e < num_entities; e++)
    for (uint i = 0; i < connections[e].size(); e++)
      this->connections[offsets[e] + i] = connections[e][i];
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
