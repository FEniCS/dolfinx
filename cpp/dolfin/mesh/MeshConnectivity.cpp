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
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
MeshConnectivity::MeshConnectivity(std::size_t d0, std::size_t d1)
    : _d0(d0), _d1(d1)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MeshConnectivity::clear()
{
  std::vector<std::int32_t>().swap(_connections);
  std::vector<std::uint32_t>().swap(_index_to_position);
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
  _index_to_position.resize(num_entities + 1);

  // Initialize data
  for (std::size_t e = 0; e < _index_to_position.size(); e++)
    _index_to_position[e] = e * num_connections;
}
//-----------------------------------------------------------------------------
void MeshConnectivity::init(std::vector<std::size_t>& num_connections)
{
  // Clear old data if any
  clear();

  // Initialize offsets and compute total size
  const std::size_t num_entities = num_connections.size();
  _index_to_position.resize(num_entities + 1);
  std::size_t size = 0;
  for (std::size_t e = 0; e < num_entities; e++)
  {
    _index_to_position[e] = size;
    size += num_connections[e];
  }
  _index_to_position[num_entities] = size;

  // Initialize connections
  _connections.resize(size);
  std::fill(_connections.begin(), _connections.end(), 0);
}
//-----------------------------------------------------------------------------
void MeshConnectivity::set(std::size_t entity, std::size_t connection,
                           std::size_t pos)
{
  assert((entity + 1) < _index_to_position.size());
  assert(pos
                < _index_to_position[entity + 1] - _index_to_position[entity]);
  _connections[_index_to_position[entity] + pos] = connection;
}
//-----------------------------------------------------------------------------
std::size_t MeshConnectivity::hash() const
{
  // Compute local hash key
  boost::hash<std::vector<std::int32_t>> uhash;
  return uhash(_connections);
}
//-----------------------------------------------------------------------------
std::string MeshConnectivity::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;
    for (std::size_t e = 0; e < _index_to_position.size() - 1; e++)
    {
      s << "  " << e << ":";
      for (std::size_t i = _index_to_position[e]; i < _index_to_position[e + 1];
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
