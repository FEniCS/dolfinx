// Copyright (C) 2006-2016 Anders Logg
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
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/utils.h>
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

  // return cell_r(*this).minCoeff();
  // double r = std::numeric_limits<double>::max();
  // for (auto& cell : MeshRange<Cell>(*this))
  //   r = std::min(r, cell.inradius());
  // return r;
}
//-----------------------------------------------------------------------------
// Create the sharing information for shared nodes
// @param mpi_comm MPI Communicator
// @param global_index List of global indices, in chunks ordered by originating
// process
// @param offsets Index into list for each process
// @return map from global index to sharing processes
std::map<std::int64_t, std::set<int>>
distribute_points_sharing(MPI_Comm mpi_comm,
                          const std::vector<std::int64_t>& global_index,
                          const std::vector<int>& offsets)
{
  const int mpi_size = dolfin::MPI::size(mpi_comm);

  // Map for sharing information
  std::map<std::int64_t, std::set<int>> point_to_procs;
  for (int i = 0; i < mpi_size; ++i)
  {
    for (int j = offsets[i]; j < offsets[i + 1]; ++j)
      point_to_procs[global_index[j]].insert(i);
  }

  std::vector<std::vector<std::int64_t>> send_sharing(mpi_size);
  std::vector<std::int64_t> recv_sharing(mpi_size);
  for (const auto& q : point_to_procs)
  {
    if (q.second.size() > 1)
    {
      for (auto r : q.second)
      {
        send_sharing[r].push_back(q.second.size() - 1);
        send_sharing[r].push_back(q.first);
        for (auto proc : q.second)
        {
          if (proc != r)
            send_sharing[r].push_back(proc);
        }
      }
    }
  }
  dolfin::MPI::all_to_all(mpi_comm, send_sharing, recv_sharing);

  // Reuse points to procs for received data
  point_to_procs.clear();
  for (auto q = recv_sharing.begin(); q < recv_sharing.end(); q += (*q + 2))
  {
    const std::int64_t global_index = *(q + 1);
    std::set<int> procs(q + 2, q + 2 + *q);
    point_to_procs.insert({global_index, procs});
  }

  return point_to_procs;
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t>
compute_global_index_set(Eigen::Ref<const EigenRowArrayXXi64> cell_nodes)
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
std::tuple<std::map<std::int64_t, std::set<int>>, dolfin::EigenRowArrayXXd>
distribute_mesh_points(MPI_Comm mpi_comm,
                       Eigen::Ref<const EigenRowArrayXXd> points,
                       const std::vector<std::int64_t>& send_global_index)
{
  std::map<std::int64_t, std::set<int>> point_to_procs;
  dolfin::EigenRowArrayXXd recv_points(send_global_index.size(), points.cols());

  int mpi_size = dolfin::MPI::size(mpi_comm);
  int mpi_rank = dolfin::MPI::rank(mpi_comm);

  // Compute where (process number) the points we need are located
  std::vector<std::int64_t> ranges(mpi_size);
  dolfin::MPI::all_gather(mpi_comm, (std::int64_t)points.rows(), ranges);
  for (std::size_t i = 1; i < ranges.size(); ++i)
    ranges[i] += ranges[i - 1];
  ranges.insert(ranges.begin(), 0);

  std::vector<int> send_offsets(mpi_size);
  std::vector<std::int64_t>::const_iterator it = send_global_index.begin();
  for (int i = 0; i < mpi_size; ++i)
  {
    // Find first index on each process
    it = std::lower_bound(it, send_global_index.end(), ranges[i]);
    send_offsets[i] = it - send_global_index.begin();
  }
  send_offsets.push_back(send_global_index.size());

  std::vector<int> send_sizes(mpi_size);
  for (int i = 0; i < mpi_size; ++i)
    send_sizes[i] = send_offsets[i + 1] - send_offsets[i];

  // Get data size to transfer in Alltoallv
  std::vector<int> recv_sizes(mpi_size);
  MPI_Alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1, MPI_INT,
               mpi_comm);

  std::vector<int> recv_offsets = {0};
  for (int i = 0; i < mpi_size; ++i)
    recv_offsets.push_back(recv_offsets.back() + recv_sizes[i]);
  std::vector<std::int64_t> recv_global_index(recv_offsets.back());

  MPI_Alltoallv(send_global_index.data(), send_sizes.data(),
                send_offsets.data(), dolfin::MPI::mpi_type<std::int64_t>(),
                recv_global_index.data(), recv_sizes.data(),
                recv_offsets.data(), dolfin::MPI::mpi_type<std::int64_t>(),
                mpi_comm);

  point_to_procs
      = distribute_points_sharing(mpi_comm, recv_global_index, recv_offsets);

  // Create compound datatype of gdim*doubles (point coords)
  MPI_Datatype compound_f64;
  MPI_Type_contiguous(points.cols(), MPI_DOUBLE, &compound_f64);
  MPI_Type_commit(&compound_f64);

  // Fill in points to send back
  dolfin::EigenRowArrayXXd send_points(recv_global_index.size(), points.cols());
  for (std::size_t i = 0; i < recv_global_index.size(); ++i)
  {
    assert(recv_global_index[i] >= ranges[mpi_rank]);
    assert(recv_global_index[i] < ranges[mpi_rank + 1]);

    int local_index = recv_global_index[i] - ranges[mpi_rank];
    send_points.row(i) = points.row(local_index);
  }

  // Get points back, matching indices in global_index_set
  MPI_Alltoallv(send_points.data(), recv_sizes.data(), recv_offsets.data(),
                compound_f64, recv_points.data(), send_sizes.data(),
                send_offsets.data(), compound_f64, mpi_comm);

  return std::make_tuple(std::move(point_to_procs), std::move(recv_points));
}
//-----------------------------------------------------------------------------
std::tuple<
    std::vector<std::int64_t>, std::map<std::int32_t, std::set<int>>,
    Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
    EigenRowArrayXXd, std::array<int, 4>>
point_distributor(MPI_Comm mpi_comm, int num_vertices_per_cell,
                  Eigen::Ref<const EigenRowArrayXXi64> cell_nodes,
                  Eigen::Ref<const EigenRowArrayXXd> points,
                  const std::vector<std::uint8_t>& cell_permutation)
{
  // Get set of global point indices, which exist on this process
  std::vector<std::int64_t> global_index_set
      = compute_global_index_set(cell_nodes);

  // Distribute points to processes that need them, and calculate
  // shared points. Points are returned in same order as in global_index_set.
  // Sharing information is global_index -> [remote sharing processes].
  std::map<std::int64_t, std::set<int>> point_to_procs;
  dolfin::EigenRowArrayXXd recv_points;
  std::tie(point_to_procs, recv_points)
      = distribute_mesh_points(mpi_comm, points, global_index_set);

  // Now figure out which points are (a) local, (b) owned+shared, (c) not
  // owned and which are vertices, and which are other points.

  int mpi_rank = dolfin::MPI::rank(mpi_comm);
  std::vector<std::int64_t> local_to_global;
  std::array<int, 4> num_vertices_local;
  // TODO: make into a function
  {
    // Classify all nodes
    std::set<std::int64_t> local_vertices;
    std::set<std::int64_t> shared_vertices;
    std::set<std::int64_t> ghost_vertices;
    std::set<std::int64_t> non_vertex_nodes;

    const std::int32_t num_cells = cell_nodes.rows();
    const std::int32_t num_nodes_per_cell = cell_nodes.cols();
    for (std::int32_t c = 0; c < num_cells; ++c)
    {
      // Loop over vertex nodes
      for (std::int32_t v = 0; v < num_vertices_per_cell; ++v)
      {
        // Get global node index
        std::int64_t q = cell_nodes(c, v);
        auto shared_it = point_to_procs.find(q);
        if (shared_it == point_to_procs.end())
          local_vertices.insert(q);
        else
        {
          // If lowest ranked sharing process is greather than this process,
          // then it is owner
          if (*(shared_it->second.begin()) > mpi_rank)
            shared_vertices.insert(q);
          else
            ghost_vertices.insert(q);
        }
      }
      // Non-vertex nodes
      for (std::int32_t v = num_vertices_per_cell; v < num_nodes_per_cell; ++v)
      {
        // Get global node index
        std::int64_t q = cell_nodes(c, v);
        non_vertex_nodes.insert(q);
      }
    }

    // Now fill local->global map and reorder received points
    local_to_global.insert(local_to_global.end(), local_vertices.begin(),
                           local_vertices.end());
    num_vertices_local[0] = local_to_global.size();
    local_to_global.insert(local_to_global.end(), shared_vertices.begin(),
                           shared_vertices.end());
    num_vertices_local[1] = local_to_global.size();
    local_to_global.insert(local_to_global.end(), ghost_vertices.begin(),
                           ghost_vertices.end());
    num_vertices_local[2] = local_to_global.size();
    local_to_global.insert(local_to_global.end(), non_vertex_nodes.begin(),
                           non_vertex_nodes.end());
    num_vertices_local[3] = local_to_global.size();
  }

  // Reverse map
  std::map<std::int64_t, std::int32_t> global_to_local;
  for (std::size_t i = 0; i < local_to_global.size(); ++i)
    global_to_local.insert({local_to_global[i], i});

  // Permute received points into local order
  dolfin::EigenRowArrayXXd local_points(recv_points.rows(), recv_points.cols());
  for (std::size_t i = 0; i < global_index_set.size(); ++i)
  {
    int local_idx = global_to_local[global_index_set[i]];
    local_points.row(local_idx) = recv_points.row(i);
  }

  // Convert cell_nodes to local indexing
  Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      cells_local(cell_nodes.rows(), cell_nodes.cols());
  for (int c = 0; c < cell_nodes.rows(); ++c)
    for (int v = 0; v < cell_nodes.cols(); ++v)
      cells_local(c, cell_permutation[v]) = global_to_local[cell_nodes(c, v)];

  // Convert sharing data to local indexing
  std::map<std::int32_t, std::set<int>> point_sharing;
  for (auto& q : point_to_procs)
    point_sharing.insert({global_to_local[q.first], q.second});

  return std::make_tuple(std::move(local_to_global), std::move(point_sharing),
                         std::move(cells_local), std::move(local_points),
                         num_vertices_local);
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
Mesh::Mesh(MPI_Comm comm, mesh::CellType type,
           const Eigen::Ref<const EigenRowArrayXXd> points,
           const Eigen::Ref<const EigenRowArrayXXi64> cells,
           const std::vector<std::int64_t>& global_cell_indices,
           const GhostMode ghost_mode, std::int32_t num_ghost_cells)
    : _cell_type(type), _degree(1), _mpi_comm(comm), _ghost_mode(ghost_mode),
      _unique_id(common::UniqueIdGenerator::id())
{
  const int tdim = mesh::cell_dim(_cell_type);
  const std::int32_t num_vertices_per_cell
      = mesh::num_cell_vertices(_cell_type);

  // Check size of global cell indices. If empty, construct later.
  if (global_cell_indices.size() > 0
      and global_cell_indices.size() != (std::size_t)cells.rows())
  {
    throw std::runtime_error(
        "Cannot create mesh. Wrong number of global cell indices");
  }

  // Permutation from VTK to DOLFIN order for cell geometric nodes
  std::vector<std::uint8_t> cell_permutation;
  if (type == mesh::CellType::quadrilateral)
  {
    // Quadrilateral cells does not follow counter clockwise
    // order (cc), but lexiographic order (LG). This breaks the assumptions
    // that the cell permutation is the same as the VTK-map.
    if (num_vertices_per_cell == cells.cols())
      cell_permutation = {0, 1, 2, 3};
    else
      throw std::runtime_error("Higher order quadrilateral not supported");
  }
  else
    cell_permutation = mesh::vtk_mapping(type, cells.cols());

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
  // cell topology using new local indices.
  std::array<int, 4> num_vertices_local;
  std::vector<std::int64_t> node_indices_global;
  Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_nodes;
  std::map<std::int32_t, std::set<std::int32_t>> nodes_shared;
  EigenRowArrayXXd points_received;

  std::tie(node_indices_global, nodes_shared, coordinate_nodes, points_received,
           num_vertices_local)
      = point_distributor(comm, num_vertices_per_cell, cells, points,
                          cell_permutation);

  _coordinate_dofs
      = std::make_unique<CoordinateDofs>(coordinate_nodes, cell_permutation);

  // Initialise geometry with global size, actual points, and local to
  // global map
  _geometry = std::make_unique<Geometry>(num_points_global, points_received,
                                         node_indices_global);

  // Get global vertex information
  std::uint64_t num_vertices_global;
  std::vector<std::int64_t> vertex_indices_global;
  std::map<std::int32_t, std::set<std::int32_t>> shared_vertices;
  if (_degree == 1)
  {
    num_vertices_global = num_points_global;
    vertex_indices_global = std::move(node_indices_global);
    shared_vertices = std::move(nodes_shared);
  }
  else
  {
    // For higher order meshes, vertices are a subset of points, so need
    // to build a global indexing for vertices
    std::tie(num_vertices_global, vertex_indices_global)
        = Partitioning::build_global_vertex_indices(
            comm, num_vertices_local[2], node_indices_global, nodes_shared);

    // FIXME: could be useful information. Where should it be kept?
    // Eliminate shared points which are not vertices
    for (auto it = nodes_shared.begin(); it != nodes_shared.end(); ++it)
      if (it->first < num_vertices_local[2])
        shared_vertices.insert(*it);
  }

  // Initialise vertex topology
  _topology = std::make_unique<Topology>(tdim, num_vertices_local[2],
                                         num_vertices_global);
  _topology->set_global_indices(0, vertex_indices_global);
  _topology->shared_entities(0) = shared_vertices;
  _topology->init_ghost(0, num_vertices_local[1]);

  // Set vertex ownership
  std::vector<int> vertex_owner;
  for (int i = num_vertices_local[1]; i < num_vertices_local[2]; ++i)
    vertex_owner.push_back(*(shared_vertices[i].begin()));
  _topology->entity_owner(0) = vertex_owner;

  // Initialise cell topology
  _topology->set_num_entities_global(tdim, num_cells_global);
  _topology->init_ghost(tdim, num_cells_local);

  // Add cells. Only copies the first few entries on each row
  // corresponding to vertices.
  auto cv = std::make_shared<Connectivity>(
      coordinate_nodes.leftCols(num_vertices_per_cell));
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

  // Compute connectivity
  Mesh* mesh = const_cast<Mesh*>(this);
  TopologyComputation::compute_entities(*mesh, dim);

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

  // Skip if already computed
  if (_topology->connectivity(d0, d1))
    return;

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
void Mesh::create_global_indices(std::size_t dim) const
{
  create_entities(dim);
  DistributedMeshTools::number_entities(*this, dim);
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
