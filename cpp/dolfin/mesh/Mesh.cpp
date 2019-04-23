// Copyright (C) 2006-2016 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Mesh.h"
#include "Cell.h"
#include "Connectivity.h"
#include "DistributedMeshTools.h"
#include "Facet.h"
#include "Geometry.h"
#include "MeshIterator.h"
#include "Partitioning.h"
#include "Topology.h"
#include "TopologyComputation.h"
#include "Vertex.h"
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/utils.h>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
Mesh::Mesh(MPI_Comm comm, mesh::CellType::Type type,
           const Eigen::Ref<const EigenRowArrayXXd> points,
           const Eigen::Ref<const EigenRowArrayXXi64> cells,
           const std::vector<std::int64_t>& global_cell_indices,
           const GhostMode ghost_mode, std::uint32_t num_ghost_cells)
    : _cell_type(mesh::CellType::create(type)),
      _coordinate_dofs(_cell_type->dim()), _degree(1), _mpi_comm(comm),
      _ghost_mode(ghost_mode), _unique_id(common::UniqueIdGenerator::id())
{
  const std::size_t tdim = _cell_type->dim();
  const std::int32_t num_vertices_per_cell = _cell_type->num_vertices();

  // Check size of global cell indices. If empty, construct later.
  if (global_cell_indices.size() > 0
      and global_cell_indices.size() != (std::size_t)cells.rows())
  {
    throw std::runtime_error(
        "Cannot create mesh. Wrong number of global cell indices");
  }

  // Permutation from VTK to DOLFIN order for cell geometric points
  // FIXME: should do this also for quad/hex
  // FIXME: remove duplication in CellType::vtk_mapping()
  std::vector<std::uint8_t> cell_permutation = {0, 1, 2, 3, 4, 5, 6, 7};

  // Infer if the mesh has P2 geometry (P1 has num_vertices_per_cell ==
  // cells.cols())
  if (num_vertices_per_cell != cells.cols())
  {
    if (type == mesh::CellType::Type::triangle and cells.cols() == 6)
    {
      _degree = 2;
      cell_permutation = {0, 1, 2, 5, 3, 4};
    }
    else if (type == mesh::CellType::Type::tetrahedron and cells.cols() == 10)
    {
      _degree = 2;
      cell_permutation = {0, 1, 2, 3, 9, 6, 8, 7, 5, 4};
    }
    else
    {
      throw std::runtime_error(
          "Mismatch between cell type and number of vertices per cell");
    }
  }

  // Get number of global points before distributing (which creates
  // duplicates)
  const std::uint64_t num_points_global = MPI::sum(comm, points.rows());

  // Number of cells, local (not ghost) and global.
  const std::int32_t num_cells = cells.rows();
  assert((std::int32_t)num_ghost_cells <= num_cells);
  const std::int32_t num_local_cells = num_cells - num_ghost_cells;
  const std::uint64_t num_cells_global = MPI::sum(comm, num_local_cells);

  // Compute point local-to-global map from global indices, and compute
  // cell topology using new local indices.
  std::int32_t num_vertices;
  std::vector<std::int64_t> global_point_indices;
  EigenRowArrayXXi32 coordinate_dofs;
  std::tie(num_vertices, global_point_indices, coordinate_dofs)
      = Partitioning::compute_point_mapping(num_vertices_per_cell, cells,
                                                cell_permutation);
  _coordinate_dofs.init(tdim, coordinate_dofs, cell_permutation);

  // Distribute the points across processes and calculate shared points
  EigenRowArrayXXd distributed_points;
  std::map<std::int32_t, std::set<std::int32_t>> shared_points;
  std::tie(distributed_points, shared_points)
      = Partitioning::distribute_points(comm, points, global_point_indices);

  // Initialise geometry with global size, actual points, and local to
  // global map
  _geometry = std::make_unique<Geometry>(num_points_global, distributed_points,
                                         global_point_indices);

  // Get global vertex information
  std::uint64_t num_vertices_global;
  std::vector<std::int64_t> global_vertex_indices;
  std::map<std::int32_t, std::set<std::int32_t>> shared_vertices;

  if (_degree == 1)
  {
    num_vertices_global = num_points_global;
    global_vertex_indices = std::move(global_point_indices);
    shared_vertices = std::move(shared_points);
  }
  else
  {
    // For higher order meshes, vertices are a subset of points, so need
    // to build a distinct global indexing for vertices.
    std::tie(num_vertices_global, global_vertex_indices)
        = Partitioning::build_global_vertex_indices(
            comm, num_vertices, global_point_indices, shared_points);
    // Eliminate shared points which are not vertices.
    // FIXME: could be useful information. Where should it be kept?
    for (auto it = shared_points.begin(); it != shared_points.end(); ++it)
      if (it->first < num_vertices)
        shared_vertices.insert(*it);
  }

  // Initialise vertex topology
  _topology
      = std::make_unique<Topology>(tdim, num_vertices, num_vertices_global);
  _topology->set_global_indices(0, global_vertex_indices);
  _topology->shared_entities(0) = shared_vertices;

  // Initialise cell topology
  _topology->set_num_entities_global(tdim, num_cells_global);
  _topology->init_ghost(tdim, num_local_cells);

  // Find the max vertex index of non-ghost cells.
  if (num_ghost_cells > 0)
  {
    const std::uint32_t max_vertex
        = coordinate_dofs.topLeftCorner(num_local_cells, num_vertices_per_cell)
              .maxCoeff();

    // Initialise number of local non-ghost vertices
    const std::uint32_t num_non_ghost_vertices = max_vertex + 1;
    _topology->init_ghost(0, num_non_ghost_vertices);
  }
  else
    _topology->init_ghost(0, num_vertices);

  // Add cells. Only copies the first few entries on each row
  // corresponding to vertices.
  auto cv = std::make_shared<Connectivity>(
      coordinate_dofs.leftCols(num_vertices_per_cell));
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
    : _cell_type(CellType::create(mesh._cell_type->cell_type())),
      _topology(new Topology(*mesh._topology)),
      _geometry(new Geometry(*mesh._geometry)),
      _coordinate_dofs(mesh._coordinate_dofs), _degree(mesh._degree),
      _mpi_comm(mesh.mpi_comm()), _ghost_mode(mesh._ghost_mode),
      _unique_id(common::UniqueIdGenerator::id())

{
  // Do nothing
}
//-----------------------------------------------------------------------------
Mesh::Mesh(Mesh&& mesh)
    : _cell_type(CellType::create(mesh._cell_type->cell_type())),
      _topology(std::move(mesh._topology)),
      _geometry(std::move(mesh._geometry)),
      _coordinate_dofs(std::move(mesh._coordinate_dofs)), _degree(mesh._degree),
      _mpi_comm(std::move(mesh._mpi_comm)),
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
Mesh& Mesh::operator=(const Mesh& mesh)
{
  // Assign data
  assert(mesh._topology);
  _topology = std::make_unique<Topology>(*mesh._topology);
  _geometry = std::make_unique<Geometry>(*mesh._geometry);
  _coordinate_dofs = mesh._coordinate_dofs;
  _degree = mesh._degree;

  if (mesh._cell_type)
    _cell_type.reset(mesh::CellType::create(mesh._cell_type->cell_type()));
  else
    _cell_type.reset();

  _ghost_mode = mesh._ghost_mode;
  _unique_id = common::UniqueIdGenerator::id();

  return *this;
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
mesh::CellType& Mesh::type()
{
  assert(_cell_type);
  return *_cell_type;
}
//-----------------------------------------------------------------------------
const mesh::CellType& Mesh::type() const
{
  assert(_cell_type);
  return *_cell_type;
}
//-----------------------------------------------------------------------------
std::size_t Mesh::init(int dim) const
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
void Mesh::init(std::size_t d0, std::size_t d1) const
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
void Mesh::init() const
{
  // Compute all entities
  for (int d = 0; d <= _topology->dim(); d++)
    init(d);

  // Compute all connectivity
  for (int d0 = 0; d0 <= _topology->dim(); d0++)
    for (int d1 = 0; d1 <= _topology->dim(); d1++)
      init(d0, d1);
}
//-----------------------------------------------------------------------------
void Mesh::init_global(std::size_t dim) const
{
  init(dim);
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
double Mesh::hmin() const
{
  double h = std::numeric_limits<double>::max();
  for (auto& cell : MeshRange<Cell>(*this))
    h = std::min(h, cell.h());

  return h;
}
//-----------------------------------------------------------------------------
double Mesh::hmax() const
{
  double h = 0.0;
  for (auto& cell : MeshRange<Cell>(*this))
    h = std::max(h, cell.h());

  return h;
}
//-----------------------------------------------------------------------------
double Mesh::rmin() const
{
  double r = std::numeric_limits<double>::max();
  for (auto& cell : MeshRange<Cell>(*this))
    r = std::min(r, cell.inradius());

  return r;
}
//-----------------------------------------------------------------------------
double Mesh::rmax() const
{
  double r = 0.0;
  for (auto& cell : MeshRange<Cell>(*this))
    r = std::max(r, cell.inradius());

  return r;
}
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
    std::string cell_type("undefined cell type");
    const int tdim = _topology->dim();
    if (_cell_type)
      cell_type = _cell_type->description(true);

    s << "<Mesh of topological dimension " << tdim << " (" << cell_type
      << ") with " << num_entities(0) << " vertices and " << num_entities(tdim)
      << " cells >";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
MPI_Comm Mesh::mpi_comm() const { return _mpi_comm.comm(); }
//-----------------------------------------------------------------------------
mesh::GhostMode Mesh::get_ghost_mode() const { return _ghost_mode; }
//-----------------------------------------------------------------------------
