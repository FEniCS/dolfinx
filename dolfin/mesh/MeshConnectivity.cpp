// Copyright (C) 2006-2007 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MeshConnectivity.h"
#include <boost/functional/hash.hpp>
#include <dolfin/log/log.h>
#include <sstream>

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
const MeshConnectivity& MeshConnectivity::
operator=(const MeshConnectivity& connectivity)
{
  // Copy data
  _d0 = connectivity._d0;
  _d1 = connectivity._d1;
  _connections = connectivity._connections;
  _num_global_connections = connectivity._num_global_connections;
  index_to_position = connectivity.index_to_position;

  return *this;
}
//-----------------------------------------------------------------------------
void MeshConnectivity::clear()
{
  std::vector<std::uint32_t>().swap(_connections);
  std::vector<std::uint32_t>().swap(index_to_position);
}
//-----------------------------------------------------------------------------
void MeshConnectivity::init(std::size_t num_entities,
                            std::size_t num_connections)
{
  // Clear old data if any
  clear();

  // Compute the total size
  const std::size_t size = num_entities * num_connections;

  // Allocate
  _connections.resize(size);
  std::fill(_connections.begin(), _connections.end(), 0);
  index_to_position.resize(num_entities + 1);

  // Initialize data
  for (std::size_t e = 0; e < index_to_position.size(); e++)
    index_to_position[e] = e * num_connections;
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
  dolfin_assert(pos
                < index_to_position[entity + 1] - index_to_position[entity]);
  _connections[index_to_position[entity] + pos] = connection;
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
  boost::hash<std::vector<std::uint32_t>> uhash;
  return uhash(_connections);
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
      for (std::size_t i = index_to_position[e]; i < index_to_position[e + 1];
           i++)
      {
        s << " " << _connections[i];
      }
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
