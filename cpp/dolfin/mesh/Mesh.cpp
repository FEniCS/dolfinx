// Copyright (C) 2006-2016 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Mesh.h"
#include "Connectivity.h"
#include "CoordinateDofs.h"
#include "DistributedMeshTools.h"
#include <dolfin/mesh/cell_types.h>
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
// Compute map from global node indices to local (contiguous) node
// indices, and remap cell node topology accordingly
//
// @param cell_vertices
//   Input cell topology (global indexing)
// @param cell_permutation
//   Permutation from VTK to DOLFIN index ordering
// @return
//   Local-to-global map for nodes (std::vector<std::int64_t>) and cell
//   nodes in local indexing (EigenRowArrayXXi32)
std::tuple<std::int32_t, std::vector<std::int64_t>, EigenRowArrayXXi32>
compute_cell_node_map(std::int32_t num_vertices_per_cell,
                      const Eigen::Ref<const EigenRowArrayXXi64>& cell_nodes,
                      const std::vector<std::uint8_t>& cell_permutation)
{
  const std::int32_t num_cells = cell_nodes.rows();
  const std::int32_t num_nodes_per_cell = cell_nodes.cols();

  // Cell points in local indexing
  EigenRowArrayXXi32 cell_nodes_local(num_cells, num_nodes_per_cell);

  // Loop over cells to build local-to-global map for (i) vertex nodes,
  // and (ii) then other nodes
  std::vector<std::int64_t> local_to_global;
  std::map<std::int64_t, std::int32_t> global_to_local;
  std::int32_t num_vertices_local = 0;
  int v0(0), v1(num_vertices_per_cell);
  for (std::int32_t pass = 0; pass < 2; ++pass)
  {
    for (std::int32_t c = 0; c < num_cells; ++c)
    {
      // Loop over cell nodes
      for (std::int32_t v = v0; v < v1; ++v)
      {
        // Get global node index
        std::int64_t q = cell_nodes(c, v);

        // Insert (global_vertex_index, local_vertex_index) into map. If
        // global index seen for first time, add to local-to-global map.
        auto map_it = global_to_local.insert({q, local_to_global.size()});
        if (map_it.second)
          local_to_global.push_back(q);

        // Set local node index in cell node list (applying permutation)
        cell_nodes_local(c, cell_permutation[v]) = map_it.first->second;
      }
    }

    // Store number of local vertices
    if (pass == 0)
      num_vertices_local = local_to_global.size();

    // Update node loop range for second pass
    v0 = num_vertices_per_cell;
    v1 = num_nodes_per_cell;
  }

  return std::make_tuple(num_vertices_local, std::move(local_to_global),
                         std::move(cell_nodes_local));
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
Mesh::Mesh(MPI_Comm comm, mesh::CellType type,
           const Eigen::Ref<const EigenRowArrayXXd> points,
           const Eigen::Ref<const EigenRowArrayXXi64> cells,
           const std::vector<std::int64_t>& global_cell_indices,
           const GhostMode ghost_mode, std::int32_t num_ghost_cells)
    : cell_type(type), _degree(1), _mpi_comm(comm), _ghost_mode(ghost_mode),
      _unique_id(common::UniqueIdGenerator::id())
{
  const int tdim = mesh::cell_dim(cell_type);
  const std::int32_t num_vertices_per_cell = mesh::num_cell_vertices(cell_type);

  // Check size of global cell indices. If empty, construct later.
  if (global_cell_indices.size() > 0
      and global_cell_indices.size() != (std::size_t)cells.rows())
  {
    throw std::runtime_error(
        "Cannot create mesh. Wrong number of global cell indices");
  }

  // Permutation from VTK to DOLFIN order for cell geometric nodes
  std::vector<std::uint8_t> cell_permutation;
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
  std::int32_t num_vertices_local;
  std::vector<std::int64_t> node_indices_global;
  Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_nodes;
  std::tie(num_vertices_local, node_indices_global, coordinate_nodes)
      = compute_cell_node_map(num_vertices_per_cell, cells, cell_permutation);
  _coordinate_dofs
      = std::make_unique<CoordinateDofs>(coordinate_nodes, cell_permutation);

  // Distribute the points across processes and calculate shared nodes
  EigenRowArrayXXd points_received;
  std::map<std::int32_t, std::set<std::int32_t>> nodes_shared;
  std::tie(points_received, nodes_shared)
      = Partitioning::distribute_points(comm, points, node_indices_global);

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
            comm, num_vertices_local, node_indices_global, nodes_shared);

    // FIXME: could be useful information. Where should it be kept?
    // Eliminate shared points which are not vertices
    for (auto it = nodes_shared.begin(); it != nodes_shared.end(); ++it)
      if (it->first < num_vertices_local)
        shared_vertices.insert(*it);
  }

  // Initialise vertex topology
  _topology = std::make_unique<Topology>(tdim, num_vertices_local,
                                         num_vertices_global);
  _topology->set_global_indices(0, vertex_indices_global);
  _topology->shared_entities(0) = shared_vertices;

  // Initialise cell topology
  _topology->set_num_entities_global(tdim, num_cells_global);
  _topology->init_ghost(tdim, num_cells_local);

  // Find the max vertex index of non-ghost cells.
  if (num_ghost_cells > 0)
  {
    const std::uint32_t max_vertex
        = coordinate_nodes.topLeftCorner(num_cells_local, num_vertices_per_cell)
              .maxCoeff();

    // Initialise number of local non-ghost vertices
    const std::uint32_t num_non_ghost_vertices = max_vertex + 1;
    _topology->init_ghost(0, num_non_ghost_vertices);
  }
  else
    _topology->init_ghost(0, num_vertices_local);

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
    : cell_type(mesh.cell_type), _topology(new Topology(*mesh._topology)),
      _geometry(new Geometry(*mesh._geometry)),
      _coordinate_dofs(new CoordinateDofs(*mesh._coordinate_dofs)),
      _degree(mesh._degree), _mpi_comm(mesh.mpi_comm()),
      _ghost_mode(mesh._ghost_mode), _unique_id(common::UniqueIdGenerator::id())

{
  // Do nothing
}
//-----------------------------------------------------------------------------
Mesh::Mesh(Mesh&& mesh)
    : cell_type(std::move(mesh.cell_type)),
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
      << mesh::to_string(cell_type) << ") with " << num_entities(0)
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
