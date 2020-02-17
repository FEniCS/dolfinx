// Copyright (C) 2006-2019 Anders Logg and Garth N. Wells
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
  const int tdim = topology.dim();
  auto c = topology.connectivity(tdim - 1, tdim);
  if (!c)
    throw std::runtime_error("Facet-cell connectivity has not been computed");

  // Get number of connected cells for each owned facet
  auto map = topology.index_map(tdim - 1);
  assert(map);
  std::vector<int> num_cells0(map->size_local(), 0);
  for (int f = 0; f < map->size_local(); ++f)
  {
    num_cells0[f] = c->num_links(f);
    assert(num_cells0[f] == 1 or num_cells0[f] == 2);
  }

  int count0 = 0;
  int count1 = 0;
  for (std::size_t i = 0; i < num_cells0.size(); ++i)
  {
    if (num_cells0[i] == 2)
      ++count0;
    if (num_cells0[i] == 1)
      ++count1;
  }
  if (MPI::rank(MPI_COMM_WORLD) == 1)
    std::cout << "Num local: " << count0 << ", " << count1 << std::endl;

  // Get number of connected cells for each ghost facet
  std::vector<int> num_cells1(map->num_ghosts(), 0);
  for (int f = 0; f < map->num_ghosts(); ++f)
  {
    num_cells1[f] = c->num_links(map->size_local() + f);

    // TEST: For facet-based ghosting, an un-owned facet should be
    // connected to only one facet
    // if (num_cells1[f] > 1)
    //   std::cout << "Problem with ghosting" << std::endl;
    // else
    //   std::cout << "Facet as expectec" << std::endl;

    assert(num_cells1[f] == 1 or num_cells1[f] == 2);
  }

  // Get data for owner from ghosts
  std::vector<std::int32_t> owned;
  map->scatter_rev(owned, num_cells1, 1, common::IndexMap::Mode::add);

  std::vector<bool> interior_facet(num_cells0.size(), false);
  for (int f = 0; f < map->size_local(); ++f)
  {
    if (MPI::rank(MPI_COMM_WORLD) == 1)
      std::cout << "owned: " << owned[f] << std::endl;

    const int num_cells = num_cells0[f] + owned[f];
    if (num_cells > 1)
      interior_facet[f] = true;
  }

  int count = 0;
  for (std::size_t i = 0; i < interior_facet.size(); ++i)
  {
    if (interior_facet[i])
      ++count;
  }
  if (MPI::rank(MPI_COMM_WORLD) == 0)
  {
    std::cout << "Num facets: " << num_cells0.size() << std::endl;
    std::cout << "Num iterior facets: " << count << std::endl;
  }
  return interior_facet;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
Topology::Topology(int dim)
    : _global_indices(dim + 1), _shared_entities(dim + 1),
      _connectivity(dim + 1, dim + 1)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
int Topology::dim() const { return _connectivity.rows() - 1; }
//-----------------------------------------------------------------------------
void Topology::set_global_indices(
    int dim, const std::vector<std::int64_t>& global_indices)
{
  assert(dim < (int)_global_indices.size());
  _global_indices[dim] = global_indices;
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
const std::vector<std::int64_t>& Topology::global_indices(int d) const
{
  assert(d < (int)_global_indices.size());
  return _global_indices[d];
}
//-----------------------------------------------------------------------------
void Topology::set_shared_entities(
    int dim, const std::map<std::int32_t, std::set<std::int32_t>>& entities)
{
  assert(dim <= this->dim());
  _shared_entities[dim] = entities;
}
//-----------------------------------------------------------------------------
const std::map<std::int32_t, std::set<std::int32_t>>&
Topology::shared_entities(int dim) const
{
  assert(dim <= this->dim());
  return _shared_entities[dim];
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

  std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
      connectivity_facet_cell = connectivity(tdim - 1, tdim);
  if (!connectivity_facet_cell)
    throw std::runtime_error("Facet-cell connectivity missing");

  assert(_index_map[dim]);
  // std::vector<bool> marker(
  //     _index_map[dim]->size_local() + _index_map[dim]->num_ghosts(), false);
  // const int num_facets
  //     = _index_map[tdim - 1]->size_local() + _index_map[tdim -
  //     1]->num_ghosts();
  std::vector<bool> marker(_index_map[dim]->size_local(), false);
  const int num_facets = _index_map[tdim - 1]->size_local();

  // Special case for facets
  assert(_interior_facets);
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

  // Iterate over all facets, selecting only those with one cell attached
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
