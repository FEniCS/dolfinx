// Copyright (C) 2006-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Topology.h"
#include "PartitioningNew.h"
#include "TopologyComputation.h"
#include "utils.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/fem/ElementDofLayout.h>
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
      _connectivity(mesh::cell_dim(type) + 1, mesh::cell_dim(type) + 1),
      _edge_reflections(0, 0), _face_reflections(0, 0), _face_rotations(0, 0),
      _facet_permutations(0, 0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
int Topology::dim() const { return _connectivity.rows() - 1; }
//-----------------------------------------------------------------------------
const std::vector<std::int64_t>& Topology::get_global_vertices_user() const
{
  return _global_user_vertices;
}
//-----------------------------------------------------------------------------
void Topology::set_global_vertices_user(
    const std::vector<std::int64_t>& indices)
{
  _global_user_vertices = indices;
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
std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
Topology::connectivity(int d0, int d1) const
{
  assert(d0 < _connectivity.rows());
  assert(d1 < _connectivity.cols());
  return _connectivity(d0, d1);
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
        s << common::indent(_connectivity(d0, d1)->str());
        s << std::endl;
      }
    }
  }
  else
    s << "<Topology of dimension " << _dim << ">";

  return s.str();
}
//-----------------------------------------------------------------------------
Eigen::Ref<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>>
Topology::get_edge_reflections() const
{
  return _edge_reflections;
}
//-----------------------------------------------------------------------------
Eigen::Ref<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>>
Topology::get_face_reflections() const
{
  return _face_reflections;
}
//-----------------------------------------------------------------------------
Eigen::Ref<const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
Topology::get_face_rotations() const
{
  return _face_rotations;
}
//-----------------------------------------------------------------------------
Eigen::Ref<const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
Topology::get_facet_permutations() const
{
  return _facet_permutations;
}
//-----------------------------------------------------------------------------
void Topology::resize_entity_permutations(std::size_t cell_count,
                                          int edges_per_cell,
                                          int faces_per_cell)
{
  if (dim() == 3)
    _facet_permutations.resize(faces_per_cell, cell_count);
  else if (dim() == 2)
    _facet_permutations.resize(edges_per_cell, cell_count);
  else if (dim() == 1)
    _facet_permutations.resize(2, cell_count);
  else if (dim() == 0)
    _facet_permutations.resize(1, cell_count);
  _facet_permutations.fill(0);

  _edge_reflections.resize(edges_per_cell, cell_count);
  _edge_reflections.fill(false);
  _face_reflections.resize(faces_per_cell, cell_count);
  _face_reflections.fill(false);
  _face_rotations.resize(faces_per_cell, cell_count);
  _face_rotations.fill(false);
}
//-----------------------------------------------------------------------------
std::size_t Topology::entity_reflection_size() const
{
  return _edge_reflections.rows();
}
//-----------------------------------------------------------------------------
void Topology::set_entity_permutation(std::size_t cell_n, int entity_dim,
                                      std::size_t entity_index,
                                      std::uint8_t rots, std::uint8_t refs)
{
  if (entity_dim == 2)
  {
    if (dim() == 3)
      _facet_permutations(entity_index, cell_n) = 2 * rots + refs;
    _face_reflections(entity_index, cell_n) = refs;
    _face_rotations(entity_index, cell_n) = rots;
  }
  else if (entity_dim == 1)
  {
    if (dim() == 2)
      _facet_permutations(entity_index, cell_n) = refs;
    _edge_reflections(entity_index, cell_n) = refs;
  }
}
//-----------------------------------------------------------------------------
mesh::CellType Topology::cell_type() const { return _cell_type; }

//-----------------------------------------------------------------------------
mesh::Topology
mesh::create_topology(MPI_Comm comm,
                      const graph::AdjacencyList<std::int64_t>& cells,
                      CellType shape)
{
  const int size = dolfinx::MPI::size(comm);

  // Assume P1 triangles for now
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> perm(5, 3);
  for (int i = 0; i < perm.rows(); ++i)
    for (int j = 0; j < perm.cols(); ++j)
      perm(i, j) = j;

  // entity_dofs = [[set([0]), set([1]), set([2])], 3 * [set()], [set()]]
  std::vector<std::vector<std::set<int>>> entity_dofs(3);
  entity_dofs[0] = {{0}, {1}, {2}};
  entity_dofs[1] = {{}, {}, {}};
  entity_dofs[2] = {{}};
  fem::ElementDofLayout layout(1, entity_dofs, {}, {}, shape, perm);

  // TODO: This step can be skipped for 'P1' elements

  // Extract topology data, e.g.just the vertices. For P1 geometry this
  // should just be the identity operator.For other elements the
  // filtered lists may have 'gaps', i.e.the indices might not be
  // contiguous.
  const graph::AdjacencyList<std::int64_t> cells_v
      = mesh::extract_topology(layout, cells);

  // Compute the destination rank for cells on this process via graph
  // partitioning
  const std::vector<int> dest = PartitioningNew::partition_cells(
      comm, size, layout.cell_type(), cells_v);

  // Distribute cells to destination rank
  const auto [my_cells, src, original_cell_index]
      = PartitioningNew::distribute(comm, cells_v, dest);

  // Build local cell-vertex connectivity, with local vertex indices
  // [0, 1, 2, ..., n), from cell-vertex connectivity using global
  // indices and get map from global vertex indices in 'cells' to the
  // local vertex indices
  auto [cells_local, local_to_global_vertices]
      = PartitioningNew::create_local_adjacency_list(my_cells);

  // Create (i) local topology object and (ii) IndexMap for cells, and
  // set cell-vertex topology
  Topology topology_local(layout.cell_type());
  const int tdim = topology_local.dim();
  auto map = std::make_shared<common::IndexMap>(comm, cells_local.num_nodes(),
                                                std::vector<std::int64_t>(), 1);
  topology_local.set_index_map(tdim, map);
  auto _cells_local
      = std::make_shared<graph::AdjacencyList<std::int32_t>>(cells_local);
  topology_local.set_connectivity(_cells_local, tdim, 0);

  const int n = local_to_global_vertices.size();
  map = std::make_shared<common::IndexMap>(comm, n, std::vector<std::int64_t>(),
                                           1);
  topology_local.set_index_map(0, map);

  // Create facets for local topology, and attach to the topology
  // object. This will be used to find possibly shared cells
  auto [cf, fv, map0]
      = TopologyComputation::compute_entities(comm, topology_local, tdim - 1);
  topology_local.set_connectivity(cf, tdim, tdim - 1);
  topology_local.set_index_map(tdim - 1, map0);
  if (fv)
    topology_local.set_connectivity(fv, tdim - 1, 0);
  auto [fc, ignore] = TopologyComputation::compute_connectivity(topology_local,
                                                                tdim - 1, tdim);
  topology_local.set_connectivity(fc, tdim - 1, tdim);

  // FIXME: This looks weird. Revise.
  // Get facets that are on the boundary of the local topology, i.e
  // are connect to one cell only
  std::vector<bool> boundary = compute_interior_facets(topology_local);
  topology_local.set_interior_facets(boundary);
  boundary = topology_local.on_boundary(tdim - 1);

  // Build distributed cell-vertex AdjacencyList, IndexMap for
  // vertices, and map from local index to old global index
  auto [cells_d, vertex_map]
      = PartitioningNew::create_distributed_adjacency_list(
          comm, topology_local, local_to_global_vertices);

  Topology topology(layout.cell_type());

  // Set vertex IndexMap, and vertex-vertex connectivity
  auto _vertex_map = std::make_shared<common::IndexMap>(std::move(vertex_map));
  topology.set_index_map(0, _vertex_map);
  auto c0 = std::make_shared<graph::AdjacencyList<std::int32_t>>(
      _vertex_map->size_local() + _vertex_map->num_ghosts());
  topology.set_connectivity(c0, 0, 0);

  // Set cell IndexMap and cell-vertex connectivity
  auto index_map_c = std::make_shared<common::IndexMap>(
      comm, cells_d.num_nodes(), std::vector<std::int64_t>(), 1);
  topology.set_index_map(tdim, index_map_c);
  auto _cells_d = std::make_shared<graph::AdjacencyList<std::int32_t>>(cells_d);
  topology.set_connectivity(_cells_d, tdim, 0);

  return topology;
}
//-----------------------------------------------------------------------------
