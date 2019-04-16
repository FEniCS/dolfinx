// Copyright (C) 2006-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MeshTopology.h"
#include "Connectivity.h"
#include <dolfin/common/utils.h>
#include <numeric>
#include <sstream>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
MeshTopology::MeshTopology(std::size_t dim)
    : _num_entities(dim + 1, 0), _ghost_offset_index(dim + 1, 0),
      _global_num_entities(dim + 1, 0), _global_indices(dim + 1),
      _connectivity(dim + 1,
                    std::vector<std::shared_ptr<Connectivity>>(dim + 1))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
int MeshTopology::dim() const { return _num_entities.size() - 1; }
//-----------------------------------------------------------------------------
std::int32_t MeshTopology::size(int dim) const
{
  if (_num_entities.empty())
    return 0;
  else
  {
    assert(dim < (int)_num_entities.size());
    return _num_entities[dim];
  }
}
//-----------------------------------------------------------------------------
std::int64_t MeshTopology::size_global(int dim) const
{
  if (_global_num_entities.empty())
    return 0;
  else
  {
    assert(dim < (int)_global_num_entities.size());
    return _global_num_entities[dim];
  }
}
//-----------------------------------------------------------------------------
std::int32_t MeshTopology::ghost_offset(int dim) const
{
  if (_ghost_offset_index.empty())
    return 0;
  else
  {
    assert(dim < (int)_ghost_offset_index.size());
    return _ghost_offset_index[dim];
  }
}
//-----------------------------------------------------------------------------
void MeshTopology::clear(int d0, int d1)
{
  assert(d0 < (int)_connectivity.size());
  assert(d1 < (int)_connectivity[d0].size());
  _connectivity[d0][d1].reset();
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
void MeshTopology::set_global_index(std::size_t dim, std::int32_t local_index,
                                    std::int64_t global_index)
{
  assert(dim < _global_indices.size());
  assert(local_index < (std::int32_t)_global_indices[dim].size());
  _global_indices[dim][local_index] = global_index;
}
//-----------------------------------------------------------------------------
const std::vector<std::int64_t>&
MeshTopology::global_indices(std::size_t d) const
{
  assert(d < _global_indices.size());
  return _global_indices[d];
}
//-----------------------------------------------------------------------------
bool MeshTopology::have_global_indices(std::size_t dim) const
{
  assert(dim < _global_indices.size());
  return !_global_indices[dim].empty();
}
//-----------------------------------------------------------------------------
bool MeshTopology::have_shared_entities(int dim) const
{
  return (_shared_entities.find(dim) != _shared_entities.end());
}
//-----------------------------------------------------------------------------
void MeshTopology::init_global_indices(std::size_t dim, std::int64_t size)
{
  assert(dim < _global_indices.size());
  _global_indices[dim] = std::vector<std::int64_t>(size, -1);
}
//-----------------------------------------------------------------------------
std::map<std::int32_t, std::set<std::int32_t>>&
MeshTopology::shared_entities(int dim)
{
  assert(dim <= this->dim());
  return _shared_entities[dim];
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t>& MeshTopology::cell_owner() { return _cell_owner; }
//-----------------------------------------------------------------------------
const std::vector<std::int32_t>& MeshTopology::cell_owner() const
{
  return _cell_owner;
}
//-----------------------------------------------------------------------------
std::shared_ptr<Connectivity> MeshTopology::connectivity(std::size_t d0,
                                                         std::size_t d1)
{
  assert(d0 < _connectivity.size());
  assert(d1 < _connectivity[d0].size());
  return _connectivity[d0][d1];
}
//-----------------------------------------------------------------------------
std::shared_ptr<const Connectivity>
MeshTopology::connectivity(std::size_t d0, std::size_t d1) const
{
  assert(d0 < _connectivity.size());
  assert(d1 < _connectivity[d0].size());
  return _connectivity[d0][d1];
}
//-----------------------------------------------------------------------------
void MeshTopology::set_connectivity(std::shared_ptr<Connectivity> c,
                                    std::size_t d0, std::size_t d1)
{
  assert(d0 < _connectivity.size());
  assert(d1 < _connectivity[d0].size());
  _connectivity[d0][d1] = c;
}
//-----------------------------------------------------------------------------
const std::map<std::int32_t, std::set<std::int32_t>>&
MeshTopology::shared_entities(int dim) const
{
  auto e = _shared_entities.find(dim);
  if (e == _shared_entities.end())
  {
    throw std::runtime_error(
        "Shared mesh entities have not been computed for dim "
        + std::to_string(dim));
  }
  return e->second;
}
//-----------------------------------------------------------------------------
size_t MeshTopology::hash() const
{
  if (!this->connectivity(dim(), 0))
    throw std::runtime_error("Connectivity has not been computed.");
  return this->connectivity(dim(), 0)->hash();
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
        if (_connectivity[d0][d1])
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
        if (!_connectivity[d0][d1])
          continue;
        s << common::indent(_connectivity[d0][d1]->str(true));
        s << std::endl;
      }
    }
  }
  else
    s << "<MeshTopology of dimension " << _dim << ">";

  return s.str();
}
//-----------------------------------------------------------------------------
