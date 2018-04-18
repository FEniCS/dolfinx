// Copyright (C) 2006-2016 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Mesh.h"
#include "Cell.h"
#include "DistributedMeshTools.h"
#include "Facet.h"
#include "MeshIterator.h"
#include "MeshPartitioning.h"
#include "TopologyComputation.h"
#include "Vertex.h"
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/utils.h>
#include <dolfin/function/Expression.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/log/log.h>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
Mesh::Mesh(MPI_Comm comm, mesh::CellType::Type type,
           const Eigen::Ref<const EigenRowArrayXXd>& points,
           const Eigen::Ref<const EigenRowArrayXXi64>& cells)
    : common::Variable("mesh", "DOLFIN mesh"),
      _cell_type(mesh::CellType::create(type)), _topology(_cell_type->dim()),
      _geometry(points), _mpi_comm(comm), _ghost_mode("none")
{
  const std::size_t tdim = _cell_type->dim();
  const std::int32_t num_vertices_per_cell = _cell_type->num_vertices();
  assert(num_vertices_per_cell == cells.cols());

  // FIXME: make a special case in serial (no mapping required)?
  // Compute vertex local-to-global map from global indices, and computed cell
  // topology using new local indices
  const auto vmap_data = MeshPartitioning::compute_vertex_mapping(comm, cells);
  const std::vector<std::int64_t>& global_vertex_indices = vmap_data.first;
  const EigenRowArrayXXi32& local_cell_vertices = vmap_data.second;

  // FIXME: Add comment ????
  const auto vdist = MeshPartitioning::distribute_vertices(
      comm, points, global_vertex_indices);
  const EigenRowArrayXXd& vertex_coordinates = vdist.first;
  const std::map<std::int32_t, std::set<std::uint32_t>>& shared_vertices
      = vdist.second;

  // FIXME: Copy data into geometry
  // const std::size_t nvals
  //     = vertex_coordinates.rows() * vertex_coordinates.cols();
  // _geometry.x().resize(nvals);
  // std::copy(vertex_coordinates.data(), vertex_coordinates.data() + nvals,
  //           _geometry.x().begin());
  _geometry.points() = vertex_coordinates;

  // Initialise vertex topology
  const std::size_t num_vertices = vertex_coordinates.rows();
  _topology.init(0, num_vertices, num_vertices);
  _topology.init_ghost(0, num_vertices);
  _topology.init_global_indices(0, num_vertices);
  for (std::size_t i = 0; i < global_vertex_indices.size(); ++i)
    _topology.set_global_index(0, i, global_vertex_indices[i]);
  _topology.shared_entities(0) = shared_vertices;

  // Initialise cell topology
  const std::size_t num_cells = local_cell_vertices.rows();
  _topology.init(tdim, num_cells, num_cells);
  _topology.init_ghost(tdim, num_cells);
  _topology.init_global_indices(tdim, num_cells);
  _topology.connectivity(tdim, 0).init(num_cells, num_vertices_per_cell);

  // Add cells
  for (std::int32_t i = 0; i != cells.rows(); ++i)
  {
    _topology.connectivity(tdim, 0).set(i, local_cell_vertices.row(i).data());
    _topology.set_global_index(tdim, i, i);
  }
}
//-----------------------------------------------------------------------------
Mesh::Mesh(const Mesh& mesh)
    : common::Variable(mesh.name(), mesh.label()),
      _cell_type(CellType::create(mesh._cell_type->cell_type())),
      _topology(mesh._topology), _geometry(mesh._geometry),
      _mpi_comm(mesh.mpi_comm()), _ghost_mode(mesh._ghost_mode)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Mesh::Mesh(Mesh&& mesh)
    : common::Variable(std::move(mesh)),
      _cell_type(CellType::create(mesh._cell_type->cell_type())),
      _topology(std::move(mesh._topology)),
      _geometry(std::move(mesh._geometry)),
      _mpi_comm(std::move(mesh._mpi_comm)),
      _ghost_mode(std::move(mesh._ghost_mode))
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
  _topology = mesh._topology;
  _geometry = mesh._geometry;
  if (mesh._cell_type)
    _cell_type.reset(mesh::CellType::create(mesh._cell_type->cell_type()));
  else
    _cell_type.reset();

  _ghost_mode = mesh._ghost_mode;

  // Rename
  rename(mesh.name(), mesh.label());

  return *this;
}
//-----------------------------------------------------------------------------
std::size_t Mesh::init(std::size_t dim) const
{
  // This function is obviously not const since it may potentially
  // compute new connectivity. However, in a sense all connectivity of
  // a mesh always exists, it just hasn't been computed yet. The
  // const_cast is also needed to allow iterators over a const Mesh to
  // create new connectivity.

  // Skip if mesh is empty
  if (num_cells() == 0)
  {
    log::warning("Mesh is empty, unable to create entities of dimension %d.",
                 dim);
    return 0;
  }

  // Skip if already computed
  if (_topology.size(dim) > 0)
    return _topology.size(dim);

  // Skip vertices and cells (should always exist)
  if (dim == 0 || dim == _topology.dim())
    return _topology.size(dim);

  // Compute connectivity
  Mesh* mesh = const_cast<Mesh*>(this);
  TopologyComputation::compute_entities(*mesh, dim);

  return _topology.size(dim);
}
//-----------------------------------------------------------------------------
void Mesh::init(std::size_t d0, std::size_t d1) const
{
  // This function is obviously not const since it may potentially
  // compute new connectivity. However, in a sense all connectivity of
  // a mesh always exists, it just hasn't been computed yet. The
  // const_cast is also needed to allow iterators over a const Mesh to
  // create new connectivity.

  // Skip if mesh is empty
  if (num_cells() == 0)
  {
    log::warning("Mesh is empty, unable to create connectivity %d --> %d.", d0,
                 d1);
    return;
  }

  // Skip if already computed
  if (!_topology.connectivity(d0, d1).empty())
    return;

  // Compute connectivity
  Mesh* mesh = const_cast<Mesh*>(this);
  TopologyComputation::compute_connectivity(*mesh, d0, d1);
}
//-----------------------------------------------------------------------------
void Mesh::init() const
{
  // Compute all entities
  for (std::size_t d = 0; d <= topology().dim(); d++)
    init(d);

  // Compute all connectivity
  for (std::size_t d0 = 0; d0 <= topology().dim(); d0++)
    for (std::size_t d1 = 0; d1 <= topology().dim(); d1++)
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
  const std::size_t D = topology().dim();
  for (std::size_t d0 = 0; d0 <= D; d0++)
  {
    for (std::size_t d1 = 0; d1 <= D; d1++)
    {
      if (!(d0 == D && d1 == 0))
        _topology.clear(d0, d1);
    }
  }
}
//-----------------------------------------------------------------------------
std::shared_ptr<geometry::BoundingBoxTree> Mesh::bounding_box_tree() const
{
  // Allocate and build tree if necessary
  if (!_tree)
  {
    _tree.reset(new geometry::BoundingBoxTree(geometry().dim()));
    _tree->build(*this, topology().dim());
  }

  return _tree;
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
  // Get local hashes
  const std::size_t kt_local = _topology.hash();
  const std::size_t kg_local = _geometry.hash();

  // Compute global hash
  const std::size_t kt = common::hash_global(_mpi_comm.comm(), kt_local);
  const std::size_t kg = common::hash_global(_mpi_comm.comm(), kg_local);

  // Compute hash based on the Cantor pairing function
  return (kt + kg) * (kt + kg + 1) / 2 + kg;
}
//-----------------------------------------------------------------------------
std::string Mesh::str(bool verbose) const
{
  std::stringstream s;
  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    s << common::indent(_geometry.str(true));
    s << common::indent(_topology.str(true));
  }
  else
  {
    std::string cell_type("undefined cell type");
    if (_cell_type)
      cell_type = _cell_type->description(true);

    s << "<Mesh of topological dimension " << topology().dim() << " ("
      << cell_type << ") with " << num_vertices() << " vertices and "
      << num_cells() << " cells >";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
std::string Mesh::ghost_mode() const
{
  assert(_ghost_mode == "none" || _ghost_mode == "shared_vertex"
         || _ghost_mode == "shared_facet");
  return _ghost_mode;
}
//-----------------------------------------------------------------------------
