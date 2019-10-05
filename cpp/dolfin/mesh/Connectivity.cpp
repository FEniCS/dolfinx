// Copyright (C) 2006-2019 Anders Logg and Garth N. Wells
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
Connectivity::Connectivity(const std::vector<std::int32_t>& connections,
                           const std::vector<std::int32_t>& positions)
    : _connections(connections.size()), _index_to_position(positions.size())
{
  assert(positions.back() == (std::int32_t)connections.size());
  for (std::size_t i = 0; i < connections.size(); ++i)
    _connections[i] = connections[i];
  for (std::size_t i = 0; i < positions.size(); ++i)
    _index_to_position[i] = positions[i];
}
//-----------------------------------------------------------------------------
Connectivity::Connectivity(
    const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>&
        connections)
    : _connections(connections.rows() * connections.cols()),
      _index_to_position(connections.rows() + 1)
{
  // NOTE: cannot directly copy data from connections because it may be
  // a view into a larger array, e.g. for non-affine cells
  Eigen::Index k = 0;
  for (Eigen::Index i = 0; i < connections.rows(); ++i)
    for (Eigen::Index j = 0; j < connections.cols(); ++j)
      _connections[k++] = connections(i, j);

  const std::int32_t num_connections_per_entity = connections.cols();
  for (Eigen::Index e = 0; e < _index_to_position.rows(); e++)
    _index_to_position[e] = e * num_connections_per_entity;
}
//-----------------------------------------------------------------------------
std::size_t Connectivity::size(std::int32_t entity) const
{
  return (entity + 1) < _index_to_position.size()
             ? _index_to_position[entity + 1] - _index_to_position[entity]
             : 0;
}
//-----------------------------------------------------------------------------
std::size_t Connectivity::size_global(std::int32_t entity) const
{
  if (_num_global_connections.size() == 0)
    return size(entity);
  else
  {
    assert(entity < _num_global_connections.size());
    return _num_global_connections[entity];
  }
}
//-----------------------------------------------------------------------------
std::int32_t* Connectivity::connections(int entity)
{
  return (entity + 1) < _index_to_position.size()
             ? &_connections[_index_to_position[entity]]
             : nullptr;
}
//-----------------------------------------------------------------------------
const std::int32_t* Connectivity::connections(int entity) const
{
  return (entity + 1) < _index_to_position.size()
             ? &_connections[_index_to_position[entity]]
             : nullptr;
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& Connectivity::connections()
{
  return _connections;
}
//-----------------------------------------------------------------------------
const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>&
Connectivity::connections() const
{
  return _connections;
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& Connectivity::entity_positions()
{
  return _index_to_position;
}
//-----------------------------------------------------------------------------
const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>&
Connectivity::entity_positions() const
{
  return _index_to_position;
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
