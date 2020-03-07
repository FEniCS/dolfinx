// Copyright (C) 2006-2019 Anders Logg, Chris Richardson, Jorgen S. Dokken
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Mesh.h"
#include "Geometry.h"
#include "Topology.h"
#include "TopologyComputation.h"
#include "utils.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/fem/DofMapBuilder.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/io/cells.h>
#include <dolfinx/mesh/cell_types.h>
#include <memory>

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
    throw std::runtime_error("Cannot compute h min/max. No cells.");

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
} // namespace

//-----------------------------------------------------------------------------
Mesh mesh::create(
    MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& cells,
    const fem::ElementDofLayout& layout,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& x)
{
  auto [topology, src, dest] = mesh::create_topology(comm, cells, layout);

  // FIXME: Figure out how to check which entities are required
  // Initialise facet for P2
  // Create local entities
  if (topology.dim() > 1)
  {
    auto [cell_entity, entity_vertex, index_map]
        = mesh::TopologyComputation::compute_entities(comm, topology, 1);
    if (cell_entity)
      topology.set_connectivity(cell_entity, topology.dim(), 1);
    if (entity_vertex)
      topology.set_connectivity(entity_vertex, 1, 0);
    if (index_map)
      topology.set_index_map(1, index_map);

    auto [cell_facet, facet_vertex, index_map1]
        = mesh::TopologyComputation::compute_entities(comm, topology,
                                                      topology.dim() - 1);
    if (cell_facet)
      topology.set_connectivity(cell_facet, topology.dim(), topology.dim() - 1);
    if (facet_vertex)
      topology.set_connectivity(facet_vertex, topology.dim() - 1, 0);
    if (index_map1)
      topology.set_index_map(topology.dim() - 1, index_map1);
  }

  const Geometry geometry
      = mesh::create_geometry(comm, topology, layout, cells, dest, src, x);

  return Mesh(comm, topology, geometry);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
Mesh::Mesh(MPI_Comm comm, const Topology& topology, const Geometry& geometry)
    : _mpi_comm(comm)
{
  _topology = std::make_unique<Topology>(topology);
  _geometry = std::make_unique<Geometry>(geometry);
}
//-----------------------------------------------------------------------------
Mesh::Mesh(
    MPI_Comm comm, mesh::CellType type,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& points,
    const Eigen::Ref<const Eigen::Array<
        std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& cells,
    const std::vector<std::int64_t>&, const GhostMode ghost_mode, std::int32_t)
    : _mpi_comm(comm), _ghost_mode(ghost_mode),
      _unique_id(common::UniqueIdGenerator::id())
{
  assert(cells.cols() > 0);
  const fem::ElementDofLayout layout = fem::geometry_layout(type, cells.cols());
  *this = mesh::create(comm, graph::AdjacencyList<std::int64_t>(cells), layout,
                       points);
}
//-----------------------------------------------------------------------------
Mesh::Mesh(const Mesh& mesh)
    : _topology(new Topology(*mesh._topology)),
      _geometry(new Geometry(*mesh._geometry)), _mpi_comm(mesh.mpi_comm()),
      _ghost_mode(mesh._ghost_mode), _unique_id(common::UniqueIdGenerator::id())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Mesh::Mesh(Mesh&& mesh)
    : _topology(std::move(mesh._topology)),
      _geometry(std::move(mesh._geometry)),
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
Mesh& Mesh::operator=(Mesh&& mesh)
{
  _topology = std::move(mesh._topology);
  _geometry = std::move(mesh._geometry);
  this->_mpi_comm = MPI_COMM_NULL;
  std::swap(this->_mpi_comm, mesh._mpi_comm);
  _ghost_mode = std::move(mesh._ghost_mode);
  _unique_id = std::move(mesh._unique_id);

  return *this;
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
  assert(map->block_size() == 1);
  return map->size_local() + map->num_ghosts();
}
//-----------------------------------------------------------------------------
std::int64_t Mesh::num_entities_global(int dim) const
{
  assert(_topology);
  assert(_topology->index_map(dim));
  assert(_topology->index_map(dim)->block_size() == 1);
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
void Mesh::create_entity_permutations() const
{
  // FIXME: This should probably be moved to topology.

  assert(_topology);
  if (_topology->entity_reflection_size() > 0)
    return;

  const int tdim = _topology->dim();
  assert(_topology->connectivity(tdim, 0));
  const int num_cells = _topology->connectivity(tdim, 0)->num_nodes();

  _topology->resize_entity_permutations(
      num_cells, cell_num_entities(_topology->cell_type(), 1),
      cell_num_entities(_topology->cell_type(), 2));

  for (int d = 0; d < tdim; ++d)
    this->create_entities(d);

  // If the cell is a triangle or tetrahedron
  if (_topology->cell_type() == CellType::triangle
      or _topology->cell_type() == CellType::tetrahedron)
  {
    for (int cell_n = 0; cell_n < num_cells; ++cell_n)
    {
      auto cell_vertices = _topology->connectivity(tdim, 0)->links(cell_n);
      for (int d = 1; d < tdim; ++d)
      {
        assert(_topology->connectivity(d, 0));
        assert(_topology->connectivity(tdim, d));
        auto cell_entities = _topology->connectivity(tdim, d)->links(cell_n);
        for (int i = 0; i < cell_num_entities(_topology->cell_type(), d); ++i)
        {
          // Get the facet
          const int sub_e_n = cell_entities[i];

          // Number of rotations and reflections to apply to the facet
          std::uint8_t rots = 0;
          std::uint8_t refs = 0;

          auto vertices = _topology->connectivity(d, 0)->links(sub_e_n);

          // If the entity is an interval, it should be oriented pointing from
          // the lowest numbered vertex to the highest numbered vertex
          if (d == 1)
          {
            // Find iterators pointing to cell vertex given a vertex on facet
            const auto it0 = std::find(
                cell_vertices.data(),
                cell_vertices.data() + cell_vertices.size(), vertices[0]);
            const auto it1 = std::find(
                cell_vertices.data(),
                cell_vertices.data() + cell_vertices.size(), vertices[1]);

            // The number of reflections
            // Comparing iterators directly instead of values they point to
            // is sufficient here
            refs = it1 < it0;
          }
          else if (d == 2)
          {
            // Orient that triangle so the the lowest numbered vertex is the
            // origin, and the next vertex anticlockwise from the lowest has a
            // lower number than the next vertex clockwise. Find the index of
            // the lowest numbered vertex
            rots = 0;

            // Store local vertex indices here
            std::array<std::size_t, 3> e_vertices;
            // Find iterators pointing to cell vertex given a vertex on facet
            for (int j = 0; j < 3; ++j)
            {
              const auto it = std::find(
                  cell_vertices.data(),
                  cell_vertices.data() + cell_vertices.size(), vertices[j]);
              // Get the actual local vertex indices
              e_vertices[j] = it - cell_vertices.data();
            }

            for (int v = 1; v < 3; ++v)
              if (e_vertices[v] < e_vertices[rots])
                rots = v;
            // pre is the number of the next vertex clockwise from the lowest
            // numbered vertex
            const int pre
                = rots == 0 ? e_vertices[3 - 1] : e_vertices[rots - 1];
            // post is the number of the next vertex anticlockwise from the
            // lowest numbered vertex
            const int post
                = rots == 3 - 1 ? e_vertices[0] : e_vertices[rots + 1];
            // The number of reflections
            refs = post > pre;
          }

          _topology->set_entity_permutation(cell_n, d, i, rots, refs);
        }
      }
    }
  }
  // If the cell is a quad, hex or interval
  else
  {
    for (int cell_n = 0; cell_n < num_cells; ++cell_n)
    {
      auto cell_vertices
          = this->topology().connectivity(tdim, 0)->links(cell_n);
      for (int d = 1; d < tdim; ++d)
      {
        assert(_topology->connectivity(d, 0));
        assert(_topology->connectivity(tdim, d));
        auto cell_entities = _topology->connectivity(tdim, d)->links(cell_n);
        for (int i = 0; i < cell_num_entities(_topology->cell_type(), d); ++i)
        {
          // Get the facet
          const int sub_e_n = cell_entities[i];

          // Number of rotations and reflections to apply to the facet
          std::uint8_t rots = 0;
          std::uint8_t refs = 0;

          auto vertices = _topology->connectivity(d, 0)->links(sub_e_n);

          // If the entity is an interval, it should be oriented pointing from
          // the lowest numbered vertex to the highest numbered vertex
          if (d == 1)
          {
            // Find iterators pointing to cell vertex given a vertex on facet
            const auto it0 = std::find(
                cell_vertices.data(),
                cell_vertices.data() + cell_vertices.size(), vertices[0]);
            const auto it1 = std::find(
                cell_vertices.data(),
                cell_vertices.data() + cell_vertices.size(), vertices[1]);

            // The number of reflections
            refs = it1 < it0;
          }
          // Triangles and quadrilaterals
          else if (d == 2)
          {
            // quadrilateral
            // Orient that quad so the the lowest numbered vertex is the origin,
            // and the next vertex anticlockwise from the lowest has a lower
            // number than the next vertex clockwise. Find the index of the
            // lowest numbered vertex
            int num_min = -1;

            // Store local vertex indices here
            std::array<std::size_t, 4> e_vertices;
            // Find iterators pointing to cell vertex given a vertex on facet
            for (int j = 0; j < 4; ++j)
            {
              const auto it = std::find(
                  cell_vertices.data(),
                  cell_vertices.data() + cell_vertices.size(), vertices[j]);
              // Get the actual local vertex indices
              e_vertices[j] = it - cell_vertices.data();
            }

            for (int v = 0; v < 4; ++v)
            {
              if (num_min == -1 || e_vertices[v] < e_vertices[num_min])
                num_min = v;
            }

            // rots is the number of rotations to get the lowest numbered vertex
            // to the origin
            rots = num_min;

            // pre is the (local) number of the next vertex clockwise from the
            // lowest numbered vertex
            int pre = 2;

            // post is the (local) number of the next vertex anticlockwise from
            // the lowest numbered vertex
            int post = 1;

            // The tensor product ordering of quads must be taken into account
            if (num_min == 1)
            {
              pre = 0;
              post = 3;
            }
            else if (num_min == 2)
            {
              pre = 3;
              post = 0;
              rots = 3;
            }
            else if (num_min == 3)
            {
              pre = 1;
              post = 2;
              rots = 2;
            }
            // The number of reflections
            refs = (e_vertices[post] > e_vertices[pre]);
          }

          _topology->set_entity_permutation(cell_n, d, i, rots, refs);
        }
      }
    }
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
