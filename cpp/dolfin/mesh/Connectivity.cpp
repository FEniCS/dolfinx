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
Connectivity::Connectivity(std::size_t num_entities,
                           std::size_t num_connections)
{
  // Compute the total size
  const std::size_t size = num_entities * num_connections;

  // Allocate
  _connections = Eigen::Array<std::int32_t, Eigen::Dynamic, 1>::Zero(size);

  // Initialize data
  _index_to_position
      = Eigen::Array<std::int32_t, Eigen::Dynamic, 1>(num_entities + 1);
  for (Eigen::Index e = 0; e < _index_to_position.size(); e++)
    _index_to_position[e] = e * num_connections;
}
//-----------------------------------------------------------------------------
Connectivity::Connectivity(const std::vector<std::int32_t>& connections,
                           const std::vector<std::int32_t>& positions)
    : _connections(connections.size()), _index_to_position(positions.size())
{
  assert(positions.size() <= connections.size());
  for (std::size_t i = 0; i < connections.size(); ++i)
    _connections[i] = connections[i];
  for (std::size_t i = 0; i < positions.size(); ++i)
    _index_to_position[i] = positions[i];
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
void Connectivity::set_global_size(
    const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& num_global_connections)
{
  assert(num_global_connections.size() == _index_to_position.size() - 1);
  _num_global_connections = num_global_connections;
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
