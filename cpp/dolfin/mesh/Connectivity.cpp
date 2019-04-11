// Copyright (C) 2006-2007 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Connectivity.h"
#include <boost/functional/hash.hpp>
#include <sstream>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
Connectivity::Connectivity()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Connectivity::Connectivity(std::size_t num_entities,
                           std::size_t num_connections)
{
  // Compute the total size
  const std::size_t size = num_entities * num_connections;

  // Allocate
  _connections = Eigen::Array<std::int32_t, Eigen::Dynamic, 1>::Zero(size);
  _index_to_position
      = Eigen::Array<std::int32_t, Eigen::Dynamic, 1>(num_entities + 1);

  // Initialize data
  for (Eigen::Index e = 0; e < _index_to_position.size(); e++)
    _index_to_position[e] = e * num_connections;
}
//-----------------------------------------------------------------------------
Connectivity::Connectivity(std::vector<std::size_t>& num_connections)
{
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
  _connections = Eigen::Array<std::int32_t, Eigen::Dynamic, 1>::Zero(size);
}
//-----------------------------------------------------------------------------
Eigen::Ref<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>
Connectivity::connections()
{
  return _connections;
}
//-----------------------------------------------------------------------------
Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>
Connectivity::connections() const
{
  return _connections;
}
//-----------------------------------------------------------------------------
Eigen::Ref<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>
Connectivity::entity_positions()
{
  return _index_to_position;
}
//-----------------------------------------------------------------------------
Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>
Connectivity::entity_positions() const
{
  return _index_to_position;
}
//-----------------------------------------------------------------------------
void Connectivity::set(std::size_t entity, std::size_t connection,
                       std::size_t pos)
{
  assert((Eigen::Index)(entity + 1) < _index_to_position.size());
  assert((std::int32_t)pos
         < _index_to_position[entity + 1] - _index_to_position[entity]);
  _connections[_index_to_position[entity] + pos] = connection;
}
//-----------------------------------------------------------------------------
void Connectivity::set(
    std::int32_t entity,
    const Eigen::Ref<const Eigen::Array<std::int32_t, 1, Eigen::Dynamic>>
        connections)
{
  assert((entity + 1) < _index_to_position.size());
  assert(connections.size()
         == _index_to_position[entity + 1] - _index_to_position[entity]);
  std::copy(connections.data(), connections.data() + connections.size(),
            _connections.data() + _index_to_position[entity]);
}
//-----------------------------------------------------------------------------
std::size_t Connectivity::hash() const
{
  return boost::hash_range(_connections.data(),
                           _connections.data() + _connections.size());
}
//-----------------------------------------------------------------------------
std::string Connectivity::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;
    for (Eigen::Index e = 0; e < _index_to_position.size() - 1; e++)
    {
      s << "  " << e << ":";
      for (std::int32_t i = _index_to_position[e];
           i < _index_to_position[e + 1]; i++)
      {
        s << " " << _connections[i];
      }
      s << std::endl;
    }
  }
  else
  {
    s << "<Connectivity of size " << _connections.size() << ">";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
