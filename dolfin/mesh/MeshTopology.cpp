// Copyright (C) 2006-2011 Anders Logg
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
// First added:  2006-05-08
// Last changed: 2014-07-02

#include "MeshTopology.h"
#include "MeshConnectivity.h"
#include <dolfin/common/utils.h>
#include <dolfin/log/log.h>
#include <numeric>
#include <sstream>

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshTopology::MeshTopology() : Variable("topology", "mesh topology")
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshTopology::MeshTopology(const MeshTopology& topology)
    : Variable("topology", "mesh topology"),
      _num_entities(topology._num_entities),
      _ghost_offset_index(topology._ghost_offset_index),
      _global_num_entities(topology._global_num_entities),
      _global_indices(topology._global_indices),
      _shared_entities(topology._shared_entities),
      _connectivity(topology._connectivity)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshTopology::~MeshTopology()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshTopology& MeshTopology::operator=(const MeshTopology& topology)
{
  // FIXME: Call copy constructor?

  // Private data
  _num_entities = topology._num_entities;
  _global_num_entities = topology._global_num_entities;
  _ghost_offset_index = topology._ghost_offset_index;
  _global_indices = topology._global_indices;
  _shared_entities = topology._shared_entities;
  _connectivity = topology._connectivity;

  return *this;
}
//-----------------------------------------------------------------------------
std::uint32_t MeshTopology::dim() const { return _num_entities.size() - 1; }
//-----------------------------------------------------------------------------
std::uint32_t MeshTopology::size(unsigned int dim) const
{
  if (_num_entities.empty())
    return 0;

  dolfin_assert(dim < _num_entities.size());
  return _num_entities[dim];
}
//-----------------------------------------------------------------------------
std::uint64_t MeshTopology::size_global(unsigned int dim) const
{
  if (_global_num_entities.empty())
    return 0;

  dolfin_assert(dim < _global_num_entities.size());
  return _global_num_entities[dim];
}
//-----------------------------------------------------------------------------
/*
std::uint32_t MeshTopology::ghost_offset(unsigned int dim) const
{
  if (_ghost_offset_index.empty())
    return 0;

  dolfin_assert(dim < _ghost_offset_index.size());
  return _ghost_offset_index[dim];
}
*/
//-----------------------------------------------------------------------------
void MeshTopology::clear(std::size_t d0, std::size_t d1)
{
  dolfin_assert(d0 < _connectivity.size());
  dolfin_assert(d1 < _connectivity[d0].size());
  _connectivity[d0][d1].clear();
}
//-----------------------------------------------------------------------------
void MeshTopology::init(std::size_t dim)
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
void MeshTopology::init(std::size_t dim, std::int32_t local_size,
                        std::int64_t global_size)
{
  dolfin_assert(dim < _num_entities.size());
  _num_entities[dim] = local_size;

  dolfin_assert(dim < _global_num_entities.size());
  _global_num_entities[dim] = global_size;

  // FIXME: Remove this when ghost/halo cells are supported
  // If mesh is local, make shared vertices empty
  if (dim == 0 && (local_size == global_size))
    shared_entities(0);
}
//-----------------------------------------------------------------------------
void MeshTopology::init_ghost(std::size_t dim, std::size_t index)
{
  dolfin_assert(dim < _ghost_offset_index.size());
  _ghost_offset_index[dim] = index;
}
//-----------------------------------------------------------------------------
void MeshTopology::init_global_indices(std::size_t dim, std::int64_t size)
{
  dolfin_assert(dim < _global_indices.size());
  _global_indices[dim] = std::vector<std::int64_t>(size, -1);
}
//-----------------------------------------------------------------------------
std::map<std::int32_t, std::set<unsigned int>>&
MeshTopology::shared_entities(unsigned int dim)
{
  dolfin_assert(dim <= this->dim());
  return _shared_entities[dim];
}
//-----------------------------------------------------------------------------
const std::map<std::int32_t, std::set<unsigned int>>&
MeshTopology::shared_entities(unsigned int dim) const
{
  auto e = _shared_entities.find(dim);
  if (e == _shared_entities.end())
  {
    dolfin_error("MeshTopology.cpp", "get shared mesh entities",
                 "Shared mesh entities have not been computed for dim %d", dim);
  }
  return e->second;
}
//-----------------------------------------------------------------------------
size_t MeshTopology::hash() const { return (*this)(dim(), 0).hash(); }
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
        s << indent(_connectivity[d0][d1].str(true));
        s << std::endl;
      }
    }
  }
  else
    s << "<MeshTopology of dimension " << _dim << ">";

  return s.str();
}
//-----------------------------------------------------------------------------
