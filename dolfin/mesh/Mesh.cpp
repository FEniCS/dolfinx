// Copyright (C) 2006-2016 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Mesh.h"
#include "Cell.h"
#include "DistributedMeshTools.h"
#include "Facet.h"
#include "LocalMeshData.h"
#include "MeshIterator.h"
#include "MeshOrdering.h"
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

//-----------------------------------------------------------------------------
Mesh::Mesh(MPI_Comm comm)
    : Variable("mesh", "DOLFIN mesh"), _ordered(false), _mpi_comm(comm),
      _ghost_mode("none")
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Mesh::Mesh(const Mesh& mesh)
    : Variable("mesh", "DOLFIN mesh"), _ordered(false),
      _mpi_comm(mesh.mpi_comm()), _ghost_mode("none")
{
  *this = mesh;
}
//-----------------------------------------------------------------------------
Mesh::Mesh(MPI_Comm comm, LocalMeshData& local_mesh_data)
    : Variable("mesh", "DOLFIN mesh"), _ordered(false), _mpi_comm(comm),
      _ghost_mode("none")
{
  const std::string ghost_mode = parameters["ghost_mode"];
  MeshPartitioning::build_distributed_mesh(*this, local_mesh_data, ghost_mode);
}
//-----------------------------------------------------------------------------
Mesh::~Mesh()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const Mesh& Mesh::operator=(const Mesh& mesh)
{
  // Assign data
  _topology = mesh._topology;
  _geometry = mesh._geometry;
  if (mesh._cell_type)
    _cell_type.reset(CellType::create(mesh._cell_type->cell_type()));
  else
    _cell_type.reset();

  _ordered = mesh._ordered;
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
    warning("Mesh is empty, unable to create entities of dimension %d.", dim);
    return 0;
  }

  // Skip if already computed
  if (_topology.size(dim) > 0)
    return _topology.size(dim);

  // Skip vertices and cells (should always exist)
  if (dim == 0 || dim == _topology.dim())
    return _topology.size(dim);

  // Check that mesh is ordered
  if (!ordered())
  {
    dolfin_error("Mesh.cpp", "initialize mesh entities",
                 "Mesh is not ordered according to the UFC numbering "
                 "convention. Consider calling mesh.order()");
  }

  // Compute connectivity
  Mesh* mesh = const_cast<Mesh*>(this);
  TopologyComputation::compute_entities(*mesh, dim);

  // Order mesh if necessary
  if (!ordered())
    mesh->order();

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
    warning("Mesh is empty, unable to create connectivity %d --> %d.", d0, d1);
    return;
  }

  // Skip if already computed
  if (!_topology(d0, d1).empty())
    return;

  // Check that mesh is ordered
  if (!ordered())
  {
    dolfin_error("Mesh.cpp", "initialize mesh connectivity",
                 "Mesh is not ordered according to the UFC numbering "
                 "convention. Consider calling mesh.order()");
  }

  // Compute connectivity
  Mesh* mesh = const_cast<Mesh*>(this);
  TopologyComputation::compute_connectivity(*mesh, d0, d1);

  // Order mesh if necessary
  if (!ordered())
    mesh->order();
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
void Mesh::order()
{
  // Order mesh
  MeshOrdering::order(*this);

  // Remember that the mesh has been ordered
  _ordered = true;
}
//-----------------------------------------------------------------------------
bool Mesh::ordered() const
{
  // Don't check if we know (or think we know) that the mesh is ordered
  if (_ordered)
    return true;

  _ordered = MeshOrdering::ordered(*this);
  return _ordered;
}
//-----------------------------------------------------------------------------
std::shared_ptr<BoundingBoxTree> Mesh::bounding_box_tree() const
{
  // Allocate and build tree if necessary
  if (!_tree)
  {
    _tree.reset(new BoundingBoxTree());
    _tree->build(*this);
  }

  return _tree;
}
//-----------------------------------------------------------------------------
double Mesh::hmin() const
{
  double h = std::numeric_limits<double>::max();
  for (auto &cell : MeshRange<Cell>(*this))
    h = std::min(h, cell.h());

  return h;
}
//-----------------------------------------------------------------------------
double Mesh::hmax() const
{
  double h = 0.0;
  for (auto &cell : MeshRange<Cell>(*this))
    h = std::max(h, cell.h());

  return h;
}
//-----------------------------------------------------------------------------
double Mesh::rmin() const
{
  double r = std::numeric_limits<double>::max();
  for (auto &cell : MeshRange<Cell>(*this))
    r = std::min(r, cell.inradius());

  return r;
}
//-----------------------------------------------------------------------------
double Mesh::rmax() const
{
  double r = 0.0;
  for (auto &cell : MeshRange<Cell>(*this))
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
  const std::size_t kt = hash_global(_mpi_comm.comm(), kt_local);
  const std::size_t kg = hash_global(_mpi_comm.comm(), kg_local);

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

    s << indent(_geometry.str(true));
    s << indent(_topology.str(true));
  }
  else
  {
    std::string cell_type("undefined cell type");
    if (_cell_type)
      cell_type = _cell_type->description(true);

    s << "<Mesh of topological dimension " << topology().dim() << " ("
      << cell_type << ") with " << num_vertices() << " vertices and "
      << num_cells() << " cells, " << (_ordered ? "ordered" : "unordered")
      << ">";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
std::string Mesh::ghost_mode() const
{
  dolfin_assert(_ghost_mode == "none" || _ghost_mode == "shared_vertex"
                || _ghost_mode == "shared_facet");
  return _ghost_mode;
}
//-----------------------------------------------------------------------------
void Mesh::create(CellType::Type type,
                  Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> geometry,
                  Eigen::Ref<const Eigen::Matrix<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> topology)
{
  // Initialise geometry
  const std::size_t gdim = geometry.cols();
  _geometry.init(gdim, 1);

  // Set cell type
  _cell_type.reset(CellType::create(type));
  const std::size_t tdim = _cell_type->dim();
  const std::int32_t nv = _cell_type->num_vertices();
  dolfin_assert(nv == topology.cols());

  // Initialize topological dimension
  _topology.init(tdim);

  _ordered = false;

  // Initialize mesh data
  // FIXME: sort out global indices for parallel
  // This method assumes it is running in serial, and
  // sets global indices accordingly.

  // Initialise vertices
  const std::size_t num_vertices = geometry.rows();

  _topology.init(0, num_vertices, num_vertices);
  _topology.init_ghost(0, num_vertices);
  _topology.init_global_indices(0, num_vertices);
  std::vector<std::size_t> num_vertex_points(1, num_vertices);
  _geometry.init_entities(num_vertex_points);

  // Initialise cells
  const std::size_t num_cells = topology.rows();
  _topology.init(tdim, num_cells, num_cells);
  _topology.init_ghost(tdim, num_cells);
  _topology.init_global_indices(tdim, num_cells);
  _topology(tdim, 0).init(num_cells, _cell_type->num_vertices());

  // Add vertices
  std::copy(geometry.data(), geometry.data() + gdim*num_vertices,
            _geometry.x().begin());
  for (std::int32_t i = 0; i != geometry.rows(); ++i)
    _topology.set_global_index(0, i, i);

  // Add cells
  for (std::int32_t i = 0; i != topology.rows(); ++i)
  {
    _topology(tdim, 0).set(i, topology.data() + i * nv);
    _topology.set_global_index(tdim, i, i);
  }
}
//-----------------------------------------------------------------------------
