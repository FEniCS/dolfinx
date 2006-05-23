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
void MeshConnectivity::set(Array<Array<uint> >& connectivity)
{
  // Delete old data if any
  clear();

  // Count the total number of connections
  _size = 0;
  num_entities = connectivity.size();
  for (uint e = 0; e < num_entities; e++)
    _size += connectivity[e].size();

  // Allocate data
  connections = new uint[_size];
  offsets = new uint[num_entities + 1];

  // Copy data
  uint offset = 0;
  for (uint e = 0; e < num_entities; e++)
  {
    offsets[e] = offset;
    for (uint c = 0; c < connectivity[e].size(); c++)
      connections[offset++] = connectivity[e][c];
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
    for (uint pos = offsets[e]; pos < offsets[e + 1]; pos++)
      cout << " " << connections[pos];
    cout << endl;
  }
}
//-----------------------------------------------------------------------------
