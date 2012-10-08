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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2006-05-09
// Last changed: 2010-11-25

#include <dolfin/log/dolfin_log.h>
#include "MeshConnectivity.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshConnectivity::MeshConnectivity(uint d0, uint d1) : d0(d0), d1(d1)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshConnectivity::MeshConnectivity(const MeshConnectivity& connectivity)
  : d0(0), d1(0)
{
  *this = connectivity;
}
//-----------------------------------------------------------------------------
MeshConnectivity::~MeshConnectivity()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const MeshConnectivity& MeshConnectivity::operator= (const MeshConnectivity& connectivity)
{
  // Clear old data if any
  clear();

  // Copy data
  d0 = connectivity.d0;
  d1 = connectivity.d1;
  connections = connectivity.connections;
  index_to_position = connectivity.index_to_position;

  return *this;
}
//-----------------------------------------------------------------------------
void MeshConnectivity::clear()
{
  connections.clear();
  index_to_position.clear();
}
//-----------------------------------------------------------------------------
void MeshConnectivity::init(uint num_entities, uint num_connections)
{
  // Clear old data if any
  clear();

  // Compute the total size
  const uint size = num_entities*num_connections;

  // Allocate data
  connections = std::vector<uint>(size, 0);
  index_to_position.resize(num_entities + 1);

  // Initialize data
  for (uint e = 0; e < index_to_position.size(); e++)
    index_to_position[e] = e*num_connections;
}
//-----------------------------------------------------------------------------
void MeshConnectivity::init(std::vector<uint>& num_connections)
{
  // Clear old data if any
  clear();

  // Initialize offsets and compute total size
  const uint num_entities = num_connections.size();
  index_to_position.resize(num_entities + 1);
  uint size = 0;
  for (uint e = 0; e < num_entities; e++)
  {
    index_to_position[e] = size;
    size += num_connections[e];
  }
  index_to_position[num_entities] = size;

  // Initialize connections
  connections = std::vector<uint>(size, 0);
}
//-----------------------------------------------------------------------------
void MeshConnectivity::set(uint entity, uint connection, uint pos)
{
  dolfin_assert((entity + 1) < index_to_position.size());
  dolfin_assert(pos < index_to_position[entity + 1] - index_to_position[entity]);
  connections[index_to_position[entity] + pos] = connection;
}
//-----------------------------------------------------------------------------
void MeshConnectivity::set(uint entity, const std::vector<uint>& connections)
{
  dolfin_assert((entity + 1) < index_to_position.size());
  dolfin_assert(connections.size() == index_to_position[entity + 1] - index_to_position[entity]);

  // Copy data
  std::copy(connections.begin(), connections.end(),
            this->connections.begin() + index_to_position[entity]);
}
//-----------------------------------------------------------------------------
void MeshConnectivity::set(uint entity, uint* connections)
{
  dolfin_assert((entity + 1) < index_to_position.size());
  dolfin_assert(connections);

  // Copy data
  const uint num_connections
    = index_to_position[entity + 1] - index_to_position[entity];
  std::copy(connections, connections + num_connections,
            this->connections.begin() + index_to_position[entity]);
}
//-----------------------------------------------------------------------------
std::string MeshConnectivity::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    for (uint e = 0; e < index_to_position.size() - 1; e++)
    {
      s << "  " << e << ":";
      for (uint i = index_to_position[e]; i < index_to_position[e + 1]; i++)
        s << " " << connections[i];
      s << std::endl;
    }
  }
  else
  {
    s << "<MeshConnectivity " << d0 << " -- " << d1 << " of size "
          << connections.size() << ">";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
