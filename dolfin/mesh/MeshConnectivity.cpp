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

#include <boost/functional/hash.hpp>
#include <dolfin/log/log.h>
#include "MeshConnectivity.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshConnectivity::MeshConnectivity(std::size_t d0, std::size_t d1)
  : _d0(d0), _d1(d1)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshConnectivity::MeshConnectivity(const MeshConnectivity& connectivity)
  : _d0(0), _d1(0)
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
  _d0 = connectivity._d0;
  _d1 = connectivity._d1;
  _connections = connectivity._connections;
  index_to_position = connectivity.index_to_position;

  return *this;
}
//-----------------------------------------------------------------------------
void MeshConnectivity::clear()
{
  _connections.clear();
  index_to_position.clear();
}
//-----------------------------------------------------------------------------
void MeshConnectivity::init(std::size_t num_entities, std::size_t num_connections)
{
  // Clear old data if any
  clear();

  // Compute the total size
  const std::size_t size = num_entities*num_connections;

  // Allocate
  _connections.resize(size);
  std::fill(_connections.begin(), _connections.end(), 0);
  index_to_position.resize(num_entities + 1);

  // Initialize data
  for (std::size_t e = 0; e < index_to_position.size(); e++)
    index_to_position[e] = e*num_connections;
}
//-----------------------------------------------------------------------------
void MeshConnectivity::init(std::vector<std::size_t>& num_connections)
{
  // Clear old data if any
  clear();

  // Initialize offsets and compute total size
  const std::size_t num_entities = num_connections.size();
  index_to_position.resize(num_entities + 1);
  std::size_t size = 0;
  for (std::size_t e = 0; e < num_entities; e++)
  {
    index_to_position[e] = size;
    size += num_connections[e];
  }
  index_to_position[num_entities] = size;

  // Initialize connections
  _connections.resize(size);
  std::fill(_connections.begin(), _connections.end(), 0);
}
//-----------------------------------------------------------------------------
void MeshConnectivity::set(std::size_t entity, std::size_t connection,
                           std::size_t pos)
{
  dolfin_assert((entity + 1) < index_to_position.size());
  dolfin_assert(pos < index_to_position[entity + 1] - index_to_position[entity]);
  _connections[index_to_position[entity] + pos] = connection;
}
//-----------------------------------------------------------------------------
void MeshConnectivity::set(std::size_t entity, const std::vector<std::size_t>& connections)
{
  dolfin_assert((entity + 1) < index_to_position.size());
  dolfin_assert(connections.size() == index_to_position[entity + 1] - index_to_position[entity]);

  // Copy data
  std::copy(connections.begin(), connections.end(),
            _connections.begin() + index_to_position[entity]);
}
//-----------------------------------------------------------------------------
void MeshConnectivity::set(std::size_t entity, std::size_t* connections)
{
  dolfin_assert((entity + 1) < index_to_position.size());
  dolfin_assert(connections);

  // Copy data
  const std::size_t num_connections
    = index_to_position[entity + 1] - index_to_position[entity];
  std::copy(connections, connections + num_connections,
            _connections.begin() + index_to_position[entity]);
}
//-----------------------------------------------------------------------------
std::size_t MeshConnectivity::hash() const
{
  // Compute local hash key
  boost::hash<std::vector<unsigned int> > uhash;
  const std::size_t local_hash = uhash(_connections);

  // Gather all hash keys
  std::vector<std::size_t> all_hashes;
  MPI::gather(local_hash, all_hashes);

  // Hash the received hash keys
  boost::hash<std::vector<size_t> > sizet_hash;
  std::size_t global_hash = sizet_hash(all_hashes);

  // Broadcast hash key
  MPI::broadcast(global_hash);

  return global_hash;
}
//-----------------------------------------------------------------------------
std::string MeshConnectivity::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    for (std::size_t e = 0; e < index_to_position.size() - 1; e++)
    {
      s << "  " << e << ":";
      for (std::size_t i = index_to_position[e]; i < index_to_position[e + 1]; i++)
        s << " " << _connections[i];
      s << std::endl;
    }
  }
  else
  {
    s << "<MeshConnectivity " << _d0 << " -- " << _d1 << " of size "
          << _connections.size() << ">";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
