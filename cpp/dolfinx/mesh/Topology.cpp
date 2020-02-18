// Copyright (C) 2006-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Topology.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <numeric>
#include <sstream>

using namespace dolfinx;
using namespace dolfinx::mesh;

//-----------------------------------------------------------------------------
std::vector<bool> mesh::compute_interior_facets(const Topology& topology)
{
  // NOTE: Getting markers for owned and unowned facets requires reverse
  // and forward scatters. It we can work only with owned facets we
  // would need only a reverse scatter.

  const int tdim = topology.dim();
  auto c = topology.connectivity(tdim - 1, tdim);
  if (!c)
    throw std::runtime_error("Facet-cell connectivity has not been computed");

  auto map = topology.index_map(tdim - 1);
  assert(map);

  // Get number of connected cells for each ghost facet
  std::vector<int> num_cells1(map->num_ghosts(), 0);
  for (int f = 0; f < map->num_ghosts(); ++f)
  {
    num_cells1[f] = c->num_links(map->size_local() + f);
    // TEST: For facet-based ghosting, an un-owned facet should be
    // connected to only one facet
    // if (num_cells1[f] > 1)
    // {
    //   throw std::runtime_error("!!!!!!!!!!");
    //   std::cout << "!!! Problem with ghosting" << std::endl;
    // }
    // else
    //   std::cout << "Facet as expected" << std::endl;
    assert(num_cells1[f] == 1 or num_cells1[f] == 2);
  }

  // Send my ghost data to owner, and receive data for my data from
  // remote ghosts
  std::vector<std::int32_t> owned;
  map->scatter_rev(owned, num_cells1, 1, common::IndexMap::Mode::add);

  // Mark owned facets that are connected to two cells
  std::vector<int> num_cells0(map->size_local(), 0);
  for (std::size_t f = 0; f < num_cells0.size(); ++f)
  {
    assert(c->num_links(f) == 1 or c->num_links(f) == 2);
    num_cells0[f] = (c->num_links(f) + owned[f]) > 1 ? 1 : 0;
  }

  // Send owned data to ghosts, and receive ghost data from owner
  const std::vector<std::int32_t> ghost_markers
      = map->scatter_fwd(num_cells0, 1);

  // Copy data, castint 1 -> true and 0 -> false
  num_cells0.insert(num_cells0.end(), ghost_markers.begin(),
                    ghost_markers.end());
  std::vector<bool> interior_facet(num_cells0.begin(), num_cells0.end());

  return interior_facet;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
Topology::Topology(mesh::CellType type)
    : _cell_type(type),
      _connectivity(mesh::cell_dim(type) + 1, mesh::cell_dim(type) + 1)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
int Topology::dim() const { return _connectivity.rows() - 1; }
//-----------------------------------------------------------------------------
void Topology::set_global_user_vertices(
    const std::vector<std::int64_t>& vertex_indices)
{
  _global_user_vertices = vertex_indices;
}
//-----------------------------------------------------------------------------
void Topology::set_index_map(int dim,
                             std::shared_ptr<const common::IndexMap> index_map)
{
  assert(dim < (int)_index_map.size());
  _index_map[dim] = index_map;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const common::IndexMap> Topology::index_map(int dim) const
{
  assert(dim < (int)_index_map.size());
  return _index_map[dim];
}
//-----------------------------------------------------------------------------
const std::vector<std::int64_t>& Topology::get_global_user_vertices() const
{
  return _global_user_vertices;
}
//-----------------------------------------------------------------------------
std::vector<bool> Topology::on_boundary(int dim) const
{
  const int tdim = this->dim();
  if (dim >= tdim or dim < 0)
  {
    throw std::runtime_error("Invalid entity dimension: "
                             + std::to_string(dim));
  }

  if (!_interior_facets)
  {
    throw std::runtime_error(
        "Facets have not been marked for interior/exterior.");
  }

  std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
      connectivity_facet_cell = connectivity(tdim - 1, tdim);
  if (!connectivity_facet_cell)
    throw std::runtime_error("Facet-cell connectivity missing");

  // TODO: figure out if we can/should make this for owned entities only
  assert(_index_map[dim]);
  std::vector<bool> marker(
      _index_map[dim]->size_local() + _index_map[dim]->num_ghosts(), false);
  const int num_facets
      = _index_map[tdim - 1]->size_local() + _index_map[tdim - 1]->num_ghosts();

  // Special case for facets
  if (dim == tdim - 1)
  {
    for (int i = 0; i < num_facets; ++i)
    {
      assert(i < (int)_interior_facets->size());
      if (!(*_interior_facets)[i])
        marker[i] = true;
    }
    return marker;
  }

  // Get connectivity from facet to entities of interest (vertices or edges)
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
      connectivity_facet_entity = connectivity(tdim - 1, dim);
  if (!connectivity_facet_entity)
    throw std::runtime_error("Facet-entity connectivity missing");

  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& fe_offsets
      = connectivity_facet_entity->offsets();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& fe_indices
      = connectivity_facet_entity->array();

  // Iterate over all facets, selecting only those with one cell
  // attached
  for (int i = 0; i < num_facets; ++i)
  {
    assert(i < (int)_interior_facets->size());
    if (!(*_interior_facets)[i])
    {
      for (int j = fe_offsets[i]; j < fe_offsets[i + 1]; ++j)
        marker[fe_indices[j]] = true;
    }
  }

  return marker;
}
//-----------------------------------------------------------------------------
std::shared_ptr<graph::AdjacencyList<std::int32_t>>
Topology::connectivity(int d0, int d1)
{
  assert(d0 < _connectivity.rows());
  assert(d1 < _connectivity.cols());
  return _connectivity(d0, d1);
}
//-----------------------------------------------------------------------------
std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
Topology::connectivity(int d0, int d1) const
{
  assert(d0 < _connectivity.rows());
  assert(d1 < _connectivity.cols());
  return _connectivity(d0, d1);
}
//-----------------------------------------------------------------------------
void Topology::set_connectivity(
    std::shared_ptr<graph::AdjacencyList<std::int32_t>> c, int d0, int d1)
{
  assert(d0 < _connectivity.rows());
  assert(d1 < _connectivity.cols());
  _connectivity(d0, d1) = c;
}
//-----------------------------------------------------------------------------
const std::vector<bool>& Topology::interior_facets() const
{
  if (!_interior_facets)
    throw std::runtime_error("Facets marker has not been computed.");
  return *_interior_facets;
}
//-----------------------------------------------------------------------------
void Topology::set_interior_facets(const std::vector<bool>& interior_facets)
{
  _interior_facets = std::make_shared<const std::vector<bool>>(interior_facets);
}
//-----------------------------------------------------------------------------
size_t Topology::hash() const
{
  if (!this->connectivity(dim(), 0))
    throw std::runtime_error("AdjacencyList has not been computed.");
  return this->connectivity(dim(), 0)->hash();
}
//-----------------------------------------------------------------------------
std::string Topology::str(bool verbose) const
{
  const int _dim = _connectivity.rows() - 1;
  std::stringstream s;
  if (verbose)
  {
    s << str(false) << std::endl << std::endl;
    s << "  Number of entities:" << std::endl << std::endl;
    for (int d = 0; d <= _dim; d++)
    {
      if (_index_map[d])
      {
        const int size
            = _index_map[d]->size_local() + _index_map[d]->num_ghosts();
        s << "    dim = " << d << ": " << size << std::endl;
      }
    }
    s << std::endl;

    s << "  Connectivity matrix:" << std::endl << std::endl;
    s << "     ";
    for (int d1 = 0; d1 <= _dim; d1++)
      s << " " << d1;
    s << std::endl;
    for (int d0 = 0; d0 <= _dim; d0++)
    {
      s << "    " << d0;
      for (int d1 = 0; d1 <= _dim; d1++)
      {
        if (_connectivity(d0, d1))
          s << " x";
        else
          s << " -";
      }
      s << std::endl;
    }
    s << std::endl;

    for (int d0 = 0; d0 <= _dim; d0++)
    {
      for (int d1 = 0; d1 <= _dim; d1++)
      {
        if (!_connectivity(d0, d1))
          continue;
        s << common::indent(_connectivity(d0, d1)->str(true));
        s << std::endl;
      }
    }
  }
  else
    s << "<Topology of dimension " << _dim << ">";

  return s.str();
}
//-----------------------------------------------------------------------------
mesh::CellType Topology::cell_type() const { return _cell_type; }
//-----------------------------------------------------------------------------
