// Copyright (C) 2006-2019 Anders Logg, Chris Richardson, Jorgen S. Dokken
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Mesh.h"
#include "CoordinateDofs.h"
#include "Geometry.h"
#include "MeshEntity.h"
#include "MeshIterator.h"
#include "Partitioning.h"
#include "Topology.h"
#include "TopologyComputation.h"
#include "utils.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/io/cells.h>
#include <dolfinx/mesh/cell_types.h>

using namespace dolfinx;
using namespace dolfinx::mesh;

namespace
{
//-----------------------------------------------------------------------------
Eigen::ArrayXd cell_h(const mesh::Mesh& mesh)
{
  const int dim = mesh.topology().dim();
  const int num_cells = mesh.num_entities(dim);
  if (num_cells == 0)
    throw std::runtime_error("Cannnot compute h min/max. No cells.");

  Eigen::ArrayXi cells(num_cells);
  std::iota(cells.data(), cells.data() + cells.size(), 0);
  return mesh::h(mesh, cells, dim);
}
//-----------------------------------------------------------------------------
Eigen::ArrayXd cell_r(const mesh::Mesh& mesh)
{
  const int dim = mesh.topology().dim();
  const int num_cells = mesh.num_entities(dim);
  if (num_cells == 0)
    throw std::runtime_error("Cannnot compute inradius min/max. No cells.");

  Eigen::ArrayXi cells(num_cells);
  std::iota(cells.data(), cells.data() + cells.size(), 0);
  return mesh::inradius(mesh, cells);
}
//-----------------------------------------------------------------------------
std::shared_ptr<common::IndexMap>
point_map_to_vertex_map(std::shared_ptr<const common::IndexMap> point_index_map,
                        const std::vector<std::int32_t>& vertices)
{
  // Compute an IndexMap for vertices, given the IndexMap for points
  // applicable to higher-order meshes where there are more points than
  // vertices.

  std::vector<std::int64_t> vertex_local(point_index_map->size_local(), -1);
  std::int64_t local_count = vertices.size();
  std::int64_t local_offset = dolfinx::MPI::global_offset(
      point_index_map->mpi_comm(), local_count, true);
  for (std::int64_t i = 0; i < local_count; ++i)
  {
    if (vertices[i] < (int)vertex_local.size())
      vertex_local[vertices[i]] = i + local_offset;
  }

  std::vector<std::int64_t> point_ghost
      = point_index_map->scatter_fwd(vertex_local, 1);

  std::vector<std::int64_t> vertex_ghost;
  for (std::int64_t idx : point_ghost)
    if (idx != -1)
      vertex_ghost.push_back(idx);

  auto vertex_index_map = std::make_shared<common::IndexMap>(
      point_index_map->mpi_comm(), local_count,
      Eigen::Map<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>(
          vertex_ghost.data(), vertex_ghost.size()),
      1);
  return vertex_index_map;
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t> compute_global_index_set(
    const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>&
        cell_nodes)
{
  // Get set of all global points needed locally
  const std::int32_t num_cells = cell_nodes.rows();
  const std::int32_t num_nodes_per_cell = cell_nodes.cols();
  std::set<std::int64_t> gi_set;
  for (std::int32_t c = 0; c < num_cells; ++c)
    for (std::int32_t v = 0; v < num_nodes_per_cell; ++v)
      gi_set.insert(cell_nodes(c, v));

  return std::vector<std::int64_t>(gi_set.begin(), gi_set.end());
}
//-----------------------------------------------------------------------------
// Get the local points.
std::tuple<
    std::shared_ptr<common::IndexMap>, std::vector<std::int64_t>,
    Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
compute_point_distribution(
    MPI_Comm mpi_comm,
    Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        cell_nodes,
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        points)
{
  // Get set of global point indices, which exist on this process
  std::vector<std::int64_t> global_index_set
      = compute_global_index_set(cell_nodes);

  // Distribute points to processes that need them, and calculate
  // IndexMap.
  const auto [point_index_map, local_to_global, points_local]
      = Partitioning::distribute_points(mpi_comm, points, global_index_set);

  // Reverse map
  std::map<std::int64_t, std::int32_t> global_to_local;
  for (std::size_t i = 0; i < local_to_global.size(); ++i)
    global_to_local.insert({local_to_global[i], i});

  // Convert cell_nodes to local indexing
  Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      cells_local(cell_nodes.rows(), cell_nodes.cols());
  for (int c = 0; c < cell_nodes.rows(); ++c)
    for (int v = 0; v < cell_nodes.cols(); ++v)
      cells_local(c, v) = global_to_local[cell_nodes(c, v)];

  return std::tuple(point_index_map, std::move(local_to_global),
                    std::move(cells_local), std::move(points_local));
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
Mesh::Mesh(
    MPI_Comm comm, mesh::CellType type,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& points,
    const Eigen::Ref<const Eigen::Array<
        std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& cells,
    const std::vector<std::int64_t>& global_cell_indices,
    const GhostMode ghost_mode, std::int32_t num_ghost_cells)
    : _degree(1), _mpi_comm(comm), _ghost_mode(ghost_mode),
      _unique_id(common::UniqueIdGenerator::id())
{
  const int tdim = mesh::cell_dim(type);

  // Check size of global cell indices. If empty, construct later.
  if (global_cell_indices.size() > 0
      and global_cell_indices.size() != (std::size_t)cells.rows())
  {
    throw std::runtime_error(
        "Cannot create mesh. Wrong number of global cell indices");
  }
  // Find degree of mesh
  // FIXME: degree should probably be in MeshGeometry
  _degree = mesh::cell_degree(type, cells.cols());

  // Get number of nodes (global)
  const std::uint64_t num_points_global = MPI::sum(comm, points.rows());

  // Number of local cells (not including ghosts)
  const std::int32_t num_cells = cells.rows();
  assert(num_ghost_cells <= num_cells);
  const std::int32_t num_cells_local = num_cells - num_ghost_cells;

  // Compute node local-to-global map from global indices, and compute
  // cell topology using new local indices
  const auto [point_index_map, node_indices_global, coordinate_nodes,
              points_received]
      = compute_point_distribution(comm, cells, points);

  _coordinate_dofs = std::make_unique<CoordinateDofs>(coordinate_nodes);

  _geometry = std::make_unique<Geometry>(num_points_global, points_received,
                                         node_indices_global);

  // Get global vertex information
  std::vector<std::int64_t> vertex_indices_global;
  std::shared_ptr<common::IndexMap> vertex_index_map;

  // Make cell to vertex connectivity
  const std::int32_t num_vertices_per_cell = mesh::num_cell_vertices(type);
  Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      vertex_cols(cells.rows(), num_vertices_per_cell);

  if (_degree == 1)
  {
    vertex_indices_global = std::move(node_indices_global);
    vertex_index_map = point_index_map;

    const std::vector<int> vertex_indices
        = mesh::cell_vertex_indices(type, cells.cols());
    for (std::int32_t i = 0; i < num_vertices_per_cell; ++i)
      vertex_cols.col(i) = coordinate_nodes.col(vertex_indices[i]);
  }
  else
  {
    // Filter out vertices
    const std::vector<int> vertex_indices
        = mesh::cell_vertex_indices(type, cells.cols());
    std::set<std::int32_t> vertex_set;
    for (int i = 0; i < coordinate_nodes.rows(); ++i)
      for (std::size_t j = 0; j < vertex_indices.size(); ++j)
        vertex_set.insert(coordinate_nodes(i, vertex_indices[j]));
    std::vector<std::int32_t> vertices(vertex_set.begin(), vertex_set.end());
    for (int i : vertices)
      vertex_indices_global.push_back(node_indices_global[i]);

    std::map<int, int> node_index_to_vertex;
    for (std::size_t i = 0; i < vertices.size(); ++i)
      node_index_to_vertex.insert({vertices[i], i});

    for (int i = 0; i < vertex_cols.rows(); ++i)
      for (std::int32_t j = 0; j < num_vertices_per_cell; ++j)
        vertex_cols(i, j)
            = node_index_to_vertex[coordinate_nodes(i, vertex_indices[j])];

    vertex_index_map = point_map_to_vertex_map(point_index_map, vertices);
  }

  // Initialise vertex topology
  _topology = std::make_unique<Topology>(type);
  _topology->set_global_user_vertices(vertex_indices_global);
  _topology->set_index_map(0, vertex_index_map);
  const std::int32_t num_vertices
      = vertex_index_map->size_local() + vertex_index_map->num_ghosts();
  auto c0 = std::make_shared<graph::AdjacencyList<std::int32_t>>(num_vertices);
  _topology->set_connectivity(c0, 0, 0);

  // Initialise cell topology
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> cell_ghosts(num_ghost_cells);
  if ((int)global_cell_indices.size() == (num_cells_local + num_ghost_cells))
  {
    std::copy(global_cell_indices.begin() + num_cells_local,
              global_cell_indices.end(), cell_ghosts.data());
  }

  auto cell_index_map = std::make_shared<common::IndexMap>(
      _mpi_comm.comm(), num_cells_local, cell_ghosts, 1);
  _topology->set_index_map(tdim, cell_index_map);

  auto cv = std::make_shared<graph::AdjacencyList<std::int32_t>>(vertex_cols);
  _topology->set_connectivity(cv, tdim, 0);

  // Global cell indices - construct if none given
  if (global_cell_indices.empty())
  {
    // FIXME: Should global_cell_indices ever be empty?
    const std::int64_t global_cell_offset
        = MPI::global_offset(comm, num_cells, true);
    std::vector<std::int64_t> global_indices(num_cells, 0);
    std::iota(global_indices.begin(), global_indices.end(), global_cell_offset);
    // _topology->set_global_indices(tdim, global_indices);
  }
  // else
  //   _topology->set_global_indices(tdim, global_cell_indices);
}
//-----------------------------------------------------------------------------
Mesh::Mesh(const Mesh& mesh)
    : _topology(new Topology(*mesh._topology)),
      _geometry(new Geometry(*mesh._geometry)),
      _coordinate_dofs(new CoordinateDofs(*mesh._coordinate_dofs)),
      _degree(mesh._degree), _mpi_comm(mesh.mpi_comm()),
      _ghost_mode(mesh._ghost_mode), _unique_id(common::UniqueIdGenerator::id())

{
  // Do nothing
}
//-----------------------------------------------------------------------------
Mesh::Mesh(Mesh&& mesh)
    : _topology(std::move(mesh._topology)),
      _geometry(std::move(mesh._geometry)),
      _coordinate_dofs(std::move(mesh._coordinate_dofs)),
      _degree(std::move(mesh._degree)), _mpi_comm(std::move(mesh._mpi_comm)),
      _ghost_mode(std::move(mesh._ghost_mode)),
      _unique_id(std::move(mesh._unique_id))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Mesh::~Mesh()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::int32_t Mesh::num_entities(int d) const
{
  assert(_topology);
  auto map = _topology->index_map(d);
  if (!map)
  {
    throw std::runtime_error("Cannot get number of mesh entities. Have not "
                             "been created for dimension "
                             + std::to_string(d) + ".");
  }
  assert(map->block_size == 1);
  return map->size_local() + map->num_ghosts();
}
//-----------------------------------------------------------------------------
std::int64_t Mesh::num_entities_global(int dim) const
{
  assert(_topology);
  assert(_topology->index_map(dim));
  assert(_topology->index_map(dim)->block_size == 1);
  return _topology->index_map(dim)->size_global();
}
//-----------------------------------------------------------------------------
Topology& Mesh::topology()
{
  assert(_topology);
  return *_topology;
}
//-----------------------------------------------------------------------------
const Topology& Mesh::topology() const
{
  assert(_topology);
  return *_topology;
}
//-----------------------------------------------------------------------------
Geometry& Mesh::geometry()
{
  assert(_geometry);
  return *_geometry;
}
//-----------------------------------------------------------------------------
const Geometry& Mesh::geometry() const
{
  assert(_geometry);
  return *_geometry;
}
//-----------------------------------------------------------------------------
std::int32_t Mesh::create_entities(int dim) const
{
  // This function is obviously not const since it may potentially
  // compute new connectivity. However, in a sense all connectivity of a
  // mesh always exists, it just hasn't been computed yet. The
  // const_cast is also needed to allow iterators over a const Mesh to
  // create new connectivity.

  assert(_topology);

  // Skip if already computed (vertices (dim=0) should always exist)
  if (_topology->connectivity(dim, 0))
    return -1;

  // Create local entities
  const auto [cell_entity, entity_vertex, index_map]
      = TopologyComputation::compute_entities(_mpi_comm.comm(), *_topology,
                                              dim);

  if (cell_entity)
    _topology->set_connectivity(cell_entity, _topology->dim(), dim);
  if (entity_vertex)
    _topology->set_connectivity(entity_vertex, dim, 0);

  if (index_map)
    _topology->set_index_map(dim, index_map);

  return index_map->size_local();
}
//-----------------------------------------------------------------------------
void Mesh::create_connectivity(int d0, int d1) const
{
  // This function is obviously not const since it may potentially
  // compute new connectivity. However, in a sense all connectivity of a
  // mesh always exists, it just hasn't been computed yet. The
  // const_cast is also needed to allow iterators over a const Mesh to
  // create new connectivity.

  // Make sure entities exist
  create_entities(d0);
  create_entities(d1);

  // Compute connectivity
  assert(_topology);
  const auto [c_d0_d1, c_d1_d0]
      = TopologyComputation::compute_connectivity(*_topology, d0, d1);

  // NOTE: that to compute the (d0, d1) connections is it sometimes
  // necessary to compute the (d1, d0) connections. We store the (d1,
  // d0) for possible later use, but there is a memory overhead if they
  // are not required. It may be better to not automatically store
  // connectivity that was not requested, but advise in a docstring the
  // most efficient order in which to call this function if several
  // connectivities are needed.

  // Attach connectivities
  Mesh* mesh = const_cast<Mesh*>(this);
  if (c_d0_d1)
    mesh->topology().set_connectivity(c_d0_d1, d0, d1);
  if (c_d1_d0)
    mesh->topology().set_connectivity(c_d1_d0, d1, d0);

  // Special facet handing
  if (d0 == (_topology->dim() - 1) and d1 == _topology->dim())
  {
    std::vector<bool> f = compute_interior_facets(*_topology);
    _topology->set_interior_facets(f);
  }
}
//-----------------------------------------------------------------------------
void Mesh::create_connectivity_all() const
{
  // Compute all entities
  for (int d = 0; d <= _topology->dim(); d++)
    create_entities(d);

  // Compute all connectivity
  for (int d0 = 0; d0 <= _topology->dim(); d0++)
    for (int d1 = 0; d1 <= _topology->dim(); d1++)
      create_connectivity(d0, d1);
}
//-----------------------------------------------------------------------------
double Mesh::hmin() const { return cell_h(*this).minCoeff(); }
//-----------------------------------------------------------------------------
double Mesh::hmax() const { return cell_h(*this).maxCoeff(); }
//-----------------------------------------------------------------------------
double Mesh::rmin() const { return cell_r(*this).minCoeff(); }
//-----------------------------------------------------------------------------
double Mesh::rmax() const { return cell_r(*this).maxCoeff(); }
//-----------------------------------------------------------------------------
std::size_t Mesh::hash() const
{
  assert(_topology);
  assert(_geometry);

  // Get local hashes
  const std::size_t kt_local = _topology->hash();
  const std::size_t kg_local = _geometry->hash();

  // Compute global hash
  const std::size_t kt = common::hash_global(_mpi_comm.comm(), kt_local);
  const std::size_t kg = common::hash_global(_mpi_comm.comm(), kg_local);

  // Compute hash based on the Cantor pairing function
  return (kt + kg) * (kt + kg + 1) / 2 + kg;
}
//-----------------------------------------------------------------------------
std::string Mesh::str(bool verbose) const
{
  assert(_geometry);
  assert(_topology);
  std::stringstream s;
  if (verbose)
  {
    s << str(false) << std::endl << std::endl;
    s << common::indent(_geometry->str(true));
    s << common::indent(_topology->str(true));
  }
  else
  {
    const int tdim = _topology->dim();
    s << "<Mesh of topological dimension " << tdim << " ("
      << mesh::to_string(_topology->cell_type()) << ") with " << num_entities(0)
      << " vertices and " << num_entities(tdim) << " cells >";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
MPI_Comm Mesh::mpi_comm() const { return _mpi_comm.comm(); }
//-----------------------------------------------------------------------------
mesh::GhostMode Mesh::get_ghost_mode() const { return _ghost_mode; }
//-----------------------------------------------------------------------------
CoordinateDofs& Mesh::coordinate_dofs()
{
  assert(_coordinate_dofs);
  return *_coordinate_dofs;
}
//-----------------------------------------------------------------------------
const CoordinateDofs& Mesh::coordinate_dofs() const
{
  assert(_coordinate_dofs);
  return *_coordinate_dofs;
}
//-----------------------------------------------------------------------------
std::int32_t Mesh::degree() const { return _degree; }
//-----------------------------------------------------------------------------
