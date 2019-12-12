// Copyright (C) 2006-2019 Anders Logg, Chris Richardson, Jorgen S. Dokken
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Mesh.h"
#include "Connectivity.h"
#include "CoordinateDofs.h"
#include "DistributedMeshTools.h"
#include "Geometry.h"
#include "MeshEntity.h"
#include "MeshIterator.h"
#include "Partitioning.h"
#include "Topology.h"
#include "TopologyComputation.h"
#include "utils.h"
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/utils.h>
#include <dolfin/io/cells.h>
#include <dolfin/mesh/cell_types.h>

using namespace dolfin;
using namespace dolfin::mesh;

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
  std::int64_t local_offset = dolfin::MPI::global_offset(
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
std::map<std::int32_t, std::set<int>>
compute_shared_from_indexmap(const common::IndexMap& index_map)
{
  // Get neighbour processes
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(index_map.mpi_comm_neighborhood(), &indegree,
                                 &outdegree, &weighted);
  assert(indegree == outdegree);
  std::vector<int> neighbours(indegree), neighbours1(indegree),
      weights(indegree), weights1(indegree);

  MPI_Dist_graph_neighbors(index_map.mpi_comm_neighborhood(), indegree,
                           neighbours.data(), weights.data(), outdegree,
                           neighbours1.data(), weights1.data());
  std::map<int, int> proc_to_neighbour;
  for (std::size_t i = 0; i < neighbours.size(); ++i)
    proc_to_neighbour[neighbours[i]] = i;

  // Simple map for locally owned, forward shared entities
  std::map<std::int32_t, std::set<int>> shared_entities
      = index_map.compute_forward_processes();

  // Ghosts are also shared, but we need to communicate to find owners
  std::vector<std::int32_t> num_sharing(index_map.size_local(), 0);
  std::vector<int> send_sizes(neighbours.size());
  for (auto q : shared_entities)
  {
    num_sharing[q.first] = q.second.size();
    for (int r : q.second)
      send_sizes[proc_to_neighbour[r]] += q.second.size();
  }
  std::vector<int> send_offsets(neighbours.size() + 1, 0);
  std::partial_sum(send_sizes.begin(), send_sizes.end(),
                   send_offsets.begin() + 1);
  std::vector<int> send_data(send_offsets.back());

  std::vector<int> temp_offsets(send_offsets);
  for (auto q : shared_entities)
  {
    for (int r : q.second)
    {
      const int np = proc_to_neighbour[r];
      std::copy(q.second.begin(), q.second.end(),
                send_data.begin() + temp_offsets[np]);
      temp_offsets[np] += q.second.size();
    }
  }

  std::vector<std::int32_t> ghost_shared_sizes
      = index_map.scatter_fwd(num_sharing, 1);

  // Count up how many to receive from each neighbour...
  std::vector<int> recv_sizes(neighbours.size());
  for (int i = 0; i < index_map.num_ghosts(); ++i)
  {
    int np = proc_to_neighbour[index_map.ghost_owners()[i]];
    recv_sizes[np] += ghost_shared_sizes[i];
  }
  std::vector<int> recv_offsets(neighbours.size() + 1, 0);
  std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                   recv_offsets.begin() + 1);
  std::vector<int> recv_data(recv_offsets.back());

  MPI_Neighbor_alltoallv(send_data.data(), send_sizes.data(),
                         send_offsets.data(), MPI_INT, recv_data.data(),
                         recv_sizes.data(), recv_offsets.data(), MPI_INT,
                         index_map.mpi_comm_neighborhood());

  int rank = dolfin::MPI::rank(index_map.mpi_comm());
  for (int i = 0; i < index_map.num_ghosts(); ++i)
  {
    const int p = index_map.ghost_owners()[i];
    const int np = proc_to_neighbour[p];
    std::set<int> sharing_set(recv_data.begin() + recv_offsets[np],
                              recv_data.begin() + recv_offsets[np]
                                  + ghost_shared_sizes[i]);
    sharing_set.insert(p);
    sharing_set.erase(rank);
    recv_offsets[np] += ghost_shared_sizes[i];
    shared_entities[i + index_map.size_local()] = sharing_set;
  }
  return shared_entities;
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
        points,
    mesh::CellType type)
{
  // Get set of global point indices, which exist on this process
  std::vector<std::int64_t> global_index_set
      = compute_global_index_set(cell_nodes);

  // Distribute points to processes that need them, and calculate
  // IndexMap.
  auto [point_index_map, local_to_global, points_local]
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
    : _cell_type(type), _degree(1), _mpi_comm(comm), _ghost_mode(ghost_mode),
      _unique_id(common::UniqueIdGenerator::id())
{
  const int tdim = mesh::cell_dim(_cell_type);

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

  // Number of cells (global)
  const std::int64_t num_cells_global = MPI::sum(comm, num_cells_local);

  // Compute node local-to-global map from global indices, and compute
  // cell topology using new local indices
  auto [point_index_map, node_indices_global, coordinate_nodes, points_received]
      = compute_point_distribution(comm, cells, points, type);

  _coordinate_dofs = std::make_unique<CoordinateDofs>(coordinate_nodes);

  _geometry = std::make_unique<Geometry>(num_points_global, points_received,
                                         node_indices_global);

  // Compute shared points from IndexMap
  std::map<std::int32_t, std::set<int>> shared_points
      = compute_shared_from_indexmap(*point_index_map);

  // Get global vertex information
  std::vector<std::int64_t> vertex_indices_global;
  std::map<std::int32_t, std::set<std::int32_t>> shared_vertices;
  std::shared_ptr<common::IndexMap> vertex_index_map;

  // Make cell to vertex connectivity
  const std::int32_t num_vertices_per_cell
      = mesh::num_cell_vertices(_cell_type);
  Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      vertex_cols(cells.rows(), num_vertices_per_cell);

  if (_degree == 1)
  {
    vertex_indices_global = std::move(node_indices_global);
    shared_vertices = std::move(shared_points);
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

    std::stringstream s;
    s << MPI::rank(comm) << " ";
    s << "V=";
    for (int q : vertices)
      s << q << " ";
    s << "\n\n";
    std::cout << s.str();

    vertex_index_map = point_map_to_vertex_map(point_index_map, vertices);

    s.str("");

    s << "Made vertex_index_map for higher order with "
      << vertex_index_map->size_local() << " local and "
      << vertex_index_map->num_ghosts() << " ghost indices\n";
    shared_vertices = compute_shared_from_indexmap(*vertex_index_map);
    for (auto q : shared_vertices)
    {
      s << q.first << ":{";
      for (int r : q.second)
        s << r << " ";
      s << "},";
    }
    std::cout << s.str() << "\n";
  }

  // Initialise vertex topology
  _topology = std::make_unique<Topology>(
      tdim, vertex_index_map->size_local() + vertex_index_map->num_ghosts(),
      vertex_index_map->size_global());
  _topology->set_global_indices(0, vertex_indices_global);
  _topology->set_shared_entities(0, shared_vertices);
  _topology->init_ghost(0, vertex_index_map->size_local());
  _topology->set_index_map(0, vertex_index_map);

  // Set vertex ownership from IndexMap (TODO: remove?)
  std::vector<int> vertex_owner(vertex_index_map->num_ghosts());
  auto owners = vertex_index_map->ghost_owners();
  std::copy(owners.data(), owners.data() + owners.size(), vertex_owner.begin());
  _topology->set_entity_owner(0, vertex_owner);

  // Initialise cell topology
  _topology->set_num_entities_global(tdim, num_cells_global);
  _topology->init_ghost(tdim, num_cells_local);

  auto cv = std::make_shared<Connectivity>(vertex_cols);

  _topology->set_connectivity(cv, tdim, 0);

  // Global cell indices - construct if none given
  if (global_cell_indices.empty())
  {
    // FIXME: Should global_cell_indices ever be empty?
    const std::int64_t global_cell_offset
        = MPI::global_offset(comm, num_cells, true);
    std::vector<std::int64_t> global_indices(num_cells, 0);
    std::iota(global_indices.begin(), global_indices.end(), global_cell_offset);
    _topology->set_global_indices(tdim, global_indices);
  }
  else
    _topology->set_global_indices(tdim, global_cell_indices);
}
//-----------------------------------------------------------------------------
Mesh::Mesh(const Mesh& mesh)
    : _cell_type(mesh._cell_type), _topology(new Topology(*mesh._topology)),
      _geometry(new Geometry(*mesh._geometry)),
      _coordinate_dofs(new CoordinateDofs(*mesh._coordinate_dofs)),
      _degree(mesh._degree), _mpi_comm(mesh.mpi_comm()),
      _ghost_mode(mesh._ghost_mode), _unique_id(common::UniqueIdGenerator::id())

{
  // Do nothing
}
//-----------------------------------------------------------------------------
Mesh::Mesh(Mesh&& mesh)
    : _cell_type(std::move(mesh._cell_type)),
      _topology(std::move(mesh._topology)),
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
  return _topology->size(d);
}
//-----------------------------------------------------------------------------
std::int64_t Mesh::num_entities_global(std::size_t dim) const
{
  assert(_topology);
  return _topology->size_global(dim);
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
std::size_t Mesh::create_entities(int dim) const
{
  // This function is obviously not const since it may potentially
  // compute new connectivity. However, in a sense all connectivity of a
  // mesh always exists, it just hasn't been computed yet. The
  // const_cast is also needed to allow iterators over a const Mesh to
  // create new connectivity.

  assert(_topology);

  // Skip if already computed (vertices (dim=0) should always exist)
  if (_topology->connectivity(dim, 0) or dim == 0)
    return _topology->size(dim);

  // Compute connectivity to vertices
  Mesh* mesh = const_cast<Mesh*>(this);

  // Create local entities
  TopologyComputation::compute_entities(*mesh, dim);
  // Number globally
  DistributedMeshTools::number_entities(*mesh, dim);

  return _topology->size(dim);
}
//-----------------------------------------------------------------------------
void Mesh::create_connectivity(std::size_t d0, std::size_t d1) const
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
  Mesh* mesh = const_cast<Mesh*>(this);
  TopologyComputation::compute_connectivity(*mesh, d0, d1);
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
void Mesh::clean()
{
  const std::size_t D = _topology->dim();
  for (std::size_t d0 = 0; d0 <= D; d0++)
  {
    for (std::size_t d1 = 0; d1 <= D; d1++)
    {
      if (!(d0 == D && d1 == 0))
        _topology->clear(d0, d1);
    }
  }
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
      << mesh::to_string(_cell_type) << ") with " << num_entities(0)
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
mesh::CellType Mesh::cell_type() const { return _cell_type; }
//-----------------------------------------------------------------------------
