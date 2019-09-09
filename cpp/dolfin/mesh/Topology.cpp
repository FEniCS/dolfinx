// Copyright (C) 2006-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Topology.h"
#include "Connectivity.h"
#include <dolfin/common/utils.h>
#include <numeric>
#include <sstream>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
Topology::Topology(std::size_t dim, std::int32_t num_vertices,
                   std::int64_t num_vertices_global)
    : _num_vertices(num_vertices), _ghost_offset_index(dim + 1, 0),
      _global_num_entities(dim + 1, -1), _global_indices(dim + 1),
      _shared_entities(dim + 1),
      _connectivity(dim + 1,
                    std::vector<std::shared_ptr<Connectivity>>(dim + 1))
{
  assert(!_global_num_entities.empty());
  _global_num_entities[0] = num_vertices_global;
}
//-----------------------------------------------------------------------------
int Topology::dim() const { return _connectivity.size() - 1; }
//-----------------------------------------------------------------------------
std::int32_t Topology::size(int dim) const
{
  if (dim == 0)
    return _num_vertices;

  assert(dim < (int)_connectivity.size());
  assert(!_connectivity[dim].empty());
  auto c = _connectivity[dim][0];
  if (!c)
  {
    throw std::runtime_error("Entities of dimension " + std::to_string(dim)
                             + " have not been created.");
  }

  return c->entity_positions().rows() - 1;
}
//-----------------------------------------------------------------------------
std::int64_t Topology::size_global(int dim) const
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
std::int32_t Topology::ghost_offset(int dim) const
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
void Topology::clear(int d0, int d1)
{
  assert(d0 < (int)_connectivity.size());
  assert(d1 < (int)_connectivity[d0].size());
  _connectivity[d0][d1].reset();
}
//-----------------------------------------------------------------------------
void Topology::set_num_entities_global(int dim, std::int64_t global_size)
{
  if (dim == 0)
  {
    throw std::runtime_error(
        "Cannot set number of global vertices post Topology creation.");
  }
  assert(dim < (int)_global_num_entities.size());
  _global_num_entities[dim] = global_size;

  // If mesh is local, make shared vertices empty
  // if (dim == 0 && (local_size == global_size))
  //   shared_entities(0);
}
//-----------------------------------------------------------------------------
void Topology::set_global_indices(
    int dim, const std::vector<std::int64_t>& global_indices)
{
  assert(dim < (int)_global_indices.size());
  _global_indices[dim] = global_indices;
}
//-----------------------------------------------------------------------------
void Topology::init_ghost(std::size_t dim, std::size_t index)
{
  assert(dim < _ghost_offset_index.size());
  _ghost_offset_index[dim] = index;
}
//-----------------------------------------------------------------------------
const std::vector<std::int64_t>& Topology::global_indices(std::size_t d) const
{
  assert(d < _global_indices.size());
  return _global_indices[d];
}
//-----------------------------------------------------------------------------
bool Topology::have_global_indices(std::size_t dim) const
{
  assert(dim < _global_indices.size());
  return !_global_indices[dim].empty();
}
//-----------------------------------------------------------------------------
std::map<std::int32_t, std::set<std::int32_t>>&
Topology::shared_entities(int dim)
{
  assert(dim <= this->dim());
  return _shared_entities[dim];
}
//-----------------------------------------------------------------------------
const std::map<std::int32_t, std::set<std::int32_t>>&
Topology::shared_entities(int dim) const
{
  assert(dim <= this->dim());
  return _shared_entities[dim];
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t>& Topology::cell_owner() { return _cell_owner; }
//-----------------------------------------------------------------------------
const std::vector<std::int32_t>& Topology::cell_owner() const
{
  return _cell_owner;
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t> Topology::surface_entities(int dim) const
{
  const int tdim = this->dim();

  if (dim >= tdim or dim < 0)
  {
    throw std::runtime_error("Invalid entity dimension: "
                             + std::to_string(dim));
  }

  std::shared_ptr<const Connectivity> connectivity_facet_cell
      = connectivity(tdim - 1, tdim);

  // Special case for facets
  if (dim == tdim - 1)
  {
    std::vector<std::int32_t> surface_facet_indices;
    for (int i = 0; i < size(tdim - 1); ++i)
    {
      if (connectivity_facet_cell->size_global(i) == 1)
        surface_facet_indices.push_back(i);
    }
    return surface_facet_indices;
  }

  // Get connectivity from facet to entities of interest (vertices or edges)
  std::shared_ptr<const Connectivity> connectivity_facet_entity
      = connectivity(tdim - 1, dim);
  assert(connectivity_facet_entity);
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& fe_offsets
      = connectivity_facet_entity->entity_positions();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& fe_indices
      = connectivity_facet_entity->connections();

  // Collect up set of surface entities
  std::set<std::int32_t> surface_entity_indices;

  for (int i = 0; i < size(tdim - 1); ++i)
    // Iterate over all facets, selecting only those with one cell attached
    for (int i = 0; i < size(tdim - 1); ++i)
    {
      if (connectivity_facet_cell->size_global(i) == 1)
      {
        for (int j = fe_offsets[i]; j < fe_offsets[i + 1]; ++j)
          surface_entity_indices.insert(fe_indices[j]);
      }
    }
  return std::vector<std::int32_t>(surface_entity_indices.begin(),
                                   surface_entity_indices.end());
}
//-----------------------------------------------------------------------------
std::shared_ptr<Connectivity> Topology::connectivity(std::size_t d0,
                                                     std::size_t d1)
{
  assert(d0 < _connectivity.size());
  assert(d1 < _connectivity[d0].size());
  return _connectivity[d0][d1];
}
//-----------------------------------------------------------------------------
std::shared_ptr<const Connectivity> Topology::connectivity(std::size_t d0,
                                                           std::size_t d1) const
{
  assert(d0 < _connectivity.size());
  assert(d1 < _connectivity[d0].size());
  return _connectivity[d0][d1];
}
//-----------------------------------------------------------------------------
void Topology::set_connectivity(std::shared_ptr<Connectivity> c, std::size_t d0,
                                std::size_t d1)
{
  assert(d0 < _connectivity.size());
  assert(d1 < _connectivity[d0].size());
  _connectivity[d0][d1] = c;
}
//-----------------------------------------------------------------------------
size_t Topology::hash() const
{
  if (!this->connectivity(dim(), 0))
    throw std::runtime_error("Connectivity has not been computed.");
  return this->connectivity(dim(), 0)->hash();
}
//-----------------------------------------------------------------------------
std::string Topology::str(bool verbose) const
{
  const std::size_t _dim = _connectivity.size() - 1;
  std::stringstream s;
  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    s << "  Number of entities:" << std::endl << std::endl;
    for (std::size_t d = 0; d <= _dim; d++)
      s << "    dim = " << d << ": " << size(d) << std::endl;
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
    s << "<Topology of dimension " << _dim << ">";

  return s.str();
}
//-----------------------------------------------------------------------------
