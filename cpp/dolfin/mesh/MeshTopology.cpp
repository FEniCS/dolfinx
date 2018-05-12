// Copyright (C) 2006-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MeshTopology.h"
#include "MeshConnectivity.h"
#include <dolfin/common/utils.h>
#include <dolfin/log/log.h>
#include <numeric>
#include <sstream>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
MeshTopology::MeshTopology(std::size_t dim)
    : common::Variable("topology")
{
  // Initialize number of mesh entities
  _num_entities = std::vector<std::int32_t>(dim + 1, 0);
  _global_num_entities = std::vector<std::int64_t>(dim + 1, 0);
  _ghost_offset_index = std::vector<std::size_t>(dim + 1, 0);

  // Initialize storage for global indices
  _global_indices.resize(dim + 1);

  // Initialize mesh connectivity
  _connectivity.resize(dim + 1);
  for (std::size_t d0 = 0; d0 <= dim; d0++)
    for (std::size_t d1 = 0; d1 <= dim; d1++)
      _connectivity[d0].push_back(MeshConnectivity(d0, d1));
}
//-----------------------------------------------------------------------------
std::uint32_t MeshTopology::dim() const { return _num_entities.size() - 1; }
//-----------------------------------------------------------------------------
std::uint32_t MeshTopology::size(std::uint32_t dim) const
{
  if (_num_entities.empty())
    return 0;

  assert(dim < _num_entities.size());
  return _num_entities[dim];
}
//-----------------------------------------------------------------------------
std::uint64_t MeshTopology::size_global(std::uint32_t dim) const
{
  if (_global_num_entities.empty())
    return 0;

  assert(dim < _global_num_entities.size());
  return _global_num_entities[dim];
}
//-----------------------------------------------------------------------------
/*
std::uint32_t MeshTopology::ghost_offset(std::uint32_t dim) const
{
  if (_ghost_offset_index.empty())
    return 0;

  assert(dim < _ghost_offset_index.size());
  return _ghost_offset_index[dim];
}
*/
//-----------------------------------------------------------------------------
void MeshTopology::clear(std::size_t d0, std::size_t d1)
{
  assert(d0 < _connectivity.size());
  assert(d1 < _connectivity[d0].size());
  _connectivity[d0][d1].clear();
}
//-----------------------------------------------------------------------------
void MeshTopology::init(std::size_t dim, std::int32_t local_size,
                        std::int64_t global_size)
{
  assert(dim < _num_entities.size());
  _num_entities[dim] = local_size;

  assert(dim < _global_num_entities.size());
  _global_num_entities[dim] = global_size;

  // If mesh is local, make shared vertices empty
  if (dim == 0 && (local_size == global_size))
    shared_entities(0);
}
//-----------------------------------------------------------------------------
void MeshTopology::init_ghost(std::size_t dim, std::size_t index)
{
  assert(dim < _ghost_offset_index.size());
  _ghost_offset_index[dim] = index;
}
//-----------------------------------------------------------------------------
void MeshTopology::init_global_indices(std::size_t dim, std::int64_t size)
{
  assert(dim < _global_indices.size());
  _global_indices[dim] = std::vector<std::int64_t>(size, -1);
}
//-----------------------------------------------------------------------------
std::map<std::int32_t, std::set<std::uint32_t>>&
MeshTopology::shared_entities(std::uint32_t dim)
{
  assert(dim <= this->dim());
  return _shared_entities[dim];
}
//-----------------------------------------------------------------------------
const std::map<std::int32_t, std::set<std::uint32_t>>&
MeshTopology::shared_entities(std::uint32_t dim) const
{
  auto e = _shared_entities.find(dim);
  if (e == _shared_entities.end())
  {
    log::dolfin_error("MeshTopology.cpp", "get shared mesh entities",
                      "Shared mesh entities have not been computed for dim %d",
                      dim);
  }
  return e->second;
}
//-----------------------------------------------------------------------------
size_t MeshTopology::hash() const
{
  return this->connectivity(dim(), 0).hash();
}
//-----------------------------------------------------------------------------
std::string MeshTopology::str(bool verbose) const
{
  const std::size_t _dim = _num_entities.size() - 1;
  std::stringstream s;
  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    s << "  Number of entities:" << std::endl << std::endl;
    for (std::size_t d = 0; d <= _dim; d++)
      s << "    dim = " << d << ": " << _num_entities[d] << std::endl;
    s << std::endl;

    s << "  Connectivity matrix:" << std::endl << std::endl;
    s << "     ";
    for (std::size_t d1 = 0; d1 <= _dim; d1++)
      s << " " << d1;
    s << std::endl;
    for (std::size_t d0 = 0; d0 <= _dim; d0++)
    {
      s << "    " << d0;
      for (std::size_t d1 = 0; d1 <= _dim; d1++)
      {
        if (!_connectivity[d0][d1].empty())
          s << " x";
        else
          s << " -";
      }
      s << std::endl;
    }
    s << std::endl;

    for (std::size_t d0 = 0; d0 <= _dim; d0++)
    {
      for (std::size_t d1 = 0; d1 <= _dim; d1++)
      {
        if (_connectivity[d0][d1].empty())
          continue;
        s << common::indent(_connectivity[d0][d1].str(true));
        s << std::endl;
      }
    }
  }
  else
    s << "<MeshTopology of dimension " << _dim << ">";

  return s.str();
}
//-----------------------------------------------------------------------------
