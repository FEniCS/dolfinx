// Copyright (C) 2006-2007 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2006-05-09
// Last changed: 2010-11-25

#include <dolfin/log/dolfin_log.h>
#include "MeshConnectivity.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshConnectivity::MeshConnectivity(uint d0, uint d1)
  : d0(d0), d1(d1), _size(0), num_entities(0), connections(0), offsets(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshConnectivity::MeshConnectivity(const MeshConnectivity& connectivity)
  : d0(d0), d1(d1), _size(0), num_entities(0), connections(0), offsets(0)
{
  *this = connectivity;
}
//-----------------------------------------------------------------------------
MeshConnectivity::~MeshConnectivity()
{
  clear();
}
//-----------------------------------------------------------------------------
const MeshConnectivity& MeshConnectivity::operator= (const MeshConnectivity& connectivity)
{
  // Clear old data if any
  clear();

  // Allocate data
  _size = connectivity._size;
  num_entities = connectivity.num_entities;
  connections = new uint[_size];
  offsets = new uint[num_entities + 1];

  // Copy data
  d0 = connectivity.d0;
  d1 = connectivity.d1;
  for (uint i = 0; i < _size; i++)
    connections[i] = connectivity.connections[i];
  if (num_entities > 0)
  {
    for (uint e = 0; e <= num_entities; e++)
      offsets[e] = connectivity.offsets[e];
  }

  return *this;
}
//-----------------------------------------------------------------------------
void MeshConnectivity::clear()
{
  _size = 0;
  num_entities = 0;

  delete [] connections;
  connections = 0;

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
void MeshConnectivity::init(std::vector<uint>& num_connections)
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
  assert(entity < num_entities);
  assert(pos < offsets[entity + 1] - offsets[entity]);

  connections[offsets[entity] + pos] = connection;
}
//-----------------------------------------------------------------------------
void MeshConnectivity::set(uint entity, const std::vector<uint>& connections)
{
  assert(entity < num_entities);
  assert(connections.size() == offsets[entity + 1] - offsets[entity]);

  // Copy data
  for (uint i = 0; i < connections.size(); i++)
    this->connections[offsets[entity] + i] = connections[i];
}
//-----------------------------------------------------------------------------
void MeshConnectivity::set(uint entity, uint* connections)
{
  assert(entity < num_entities);
  assert(connections);

  // Copy data
  const uint num_connections = offsets[entity + 1] - offsets[entity];
  for (uint i = 0; i < num_connections; i++)
    this->connections[offsets[entity] + i] = connections[i];
}
//-----------------------------------------------------------------------------
void MeshConnectivity::set(const std::vector<std::vector<uint> >& connections)
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
    for (uint i = 0; i < connections[e].size(); i++)
      this->connections[offsets[e] + i] = connections[e][i];
}
//-----------------------------------------------------------------------------
std::string MeshConnectivity::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    for (uint e = 0; e < num_entities; e++)
    {
      s << "  " << e << ":";
      for (uint i = offsets[e]; i < offsets[e + 1]; i++)
        s << " " << connections[i];
      s << std::endl;
    }
  }
  else
  {
    s << "<MeshConnectivity " << d0 << " -- " << d1 << " of size " << _size << ">";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
