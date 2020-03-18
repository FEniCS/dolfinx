// Copyright (C) 2006-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Topology.h"
#include "Partitioning.h"
#include "TopologyComputation.h"
#include "utils.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/Partitioning.h>
#include <numeric>

using namespace dolfinx;
using namespace dolfinx::mesh;

namespace
{
std::pair<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>,
          Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
compute_face_permutations_simplex(
    const graph::AdjacencyList<std::int32_t>& c_to_v,
    const graph::AdjacencyList<std::int32_t>& c_to_f,
    const graph::AdjacencyList<std::int32_t>& f_to_v, int faces_per_cell)
{
  Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> reflections(
      faces_per_cell, c_to_v.num_nodes());
  Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> rotations(
      faces_per_cell, c_to_v.num_nodes());
  for (int c = 0; c < c_to_v.num_nodes(); ++c)
  {
    auto cell_vertices = c_to_v.links(c);
    auto cell_faces = c_to_f.links(c);
    for (int i = 0; i < faces_per_cell; ++i)
    {
      // Get the face
      const int face = cell_faces[i];
      auto vertices = f_to_v.links(face);

      // Orient that triangle so the the lowest numbered vertex is
      // the origin, and the next vertex anticlockwise from the
      // lowest has a lower number than the next vertex clockwise.
      // Find the index of the lowest numbered vertex

      // Store local vertex indices here
      std::array<std::int32_t, 3> e_vertices;

      // Find iterators pointing to cell vertex given a vertex on
      // facet
      for (int j = 0; j < 3; ++j)
      {
        const auto it = std::find(cell_vertices.data(),
                                  cell_vertices.data() + cell_vertices.size(),
                                  vertices[j]);
        // Get the actual local vertex indices
        e_vertices[j] = it - cell_vertices.data();
      }

      // Number of rotations
      std::uint8_t rots = 0;
      for (int v = 1; v < 3; ++v)
        if (e_vertices[v] < e_vertices[rots])
          rots = v;

      // pre is the number of the next vertex clockwise from the
      // lowest numbered vertex
      const int pre = rots == 0 ? e_vertices[3 - 1] : e_vertices[rots - 1];

      // post is the number of the next vertex anticlockwise from
      // the lowest numbered vertex
      const int post = rots == 3 - 1 ? e_vertices[0] : e_vertices[rots + 1];

      // The number of reflections
      const std::uint8_t refs = post > pre;

      reflections(i, c) = refs;
      rotations(i, c) = rots;
    }
  }

  return {std::move(reflections), std::move(rotations)};
}
//-----------------------------------------------------------------------------
std::pair<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>,
          Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
compute_face_permutations_tp(const graph::AdjacencyList<std::int32_t>& c_to_v,
                             const graph::AdjacencyList<std::int32_t>& c_to_f,
                             const graph::AdjacencyList<std::int32_t>& f_to_v,
                             int faces_per_cell)
{
  Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> reflections(
      faces_per_cell, c_to_v.num_nodes());
  Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> rotations(
      faces_per_cell, c_to_v.num_nodes());
  for (int c = 0; c < c_to_v.num_nodes(); ++c)
  {
    auto cell_vertices = c_to_v.links(c);
    auto cell_faces = c_to_f.links(c);
    for (int i = 0; i < faces_per_cell; ++i)
    {
      const int face = cell_faces[i];
      auto vertices = f_to_v.links(face);

      // quadrilateral
      // Orient that quad so the the lowest numbered vertex is the origin,
      // and the next vertex anticlockwise from the lowest has a lower
      // number than the next vertex clockwise. Find the index of the
      // lowest numbered vertex
      int num_min = -1;

      // Store local vertex indices here
      std::array<std::int32_t, 4> e_vertices;
      // Find iterators pointing to cell vertex given a vertex on facet
      for (int j = 0; j < 4; ++j)
      {
        const auto it = std::find(cell_vertices.data(),
                                  cell_vertices.data() + cell_vertices.size(),
                                  vertices[j]);
        // Get the actual local vertex indices
        e_vertices[j] = it - cell_vertices.data();
      }

      for (int v = 0; v < 4; ++v)
      {
        if (num_min == -1 or e_vertices[v] < e_vertices[num_min])
          num_min = v;
      }

      // rots is the number of rotations to get the lowest numbered vertex
      // to the origin
      std::uint8_t rots = num_min;

      // pre is the (local) number of the next vertex clockwise from the
      // lowest numbered vertex
      int pre = 2;

      // post is the (local) number of the next vertex anticlockwise from
      // the lowest numbered vertex
      int post = 1;

      // The tensor product ordering of quads must be taken into account
      assert(num_min < 4);
      switch (num_min)
      {
      case 1:
        pre = 0;
        post = 3;
        break;
      case 2:
        pre = 3;
        post = 0;
        rots = 3;
        break;
      case 3:
        pre = 1;
        post = 2;
        rots = 2;
        break;
      }

      // The number of reflections
      const std::uint8_t refs = (e_vertices[post] > e_vertices[pre]);

      reflections(i, c) = refs;
      rotations(i, c) = rots;
    }
  }

  return {std::move(reflections), std::move(rotations)};
}
//-----------------------------------------------------------------------------
std::unordered_map<std::int64_t, std::vector<int>>
compute_sharing(MPI_Comm comm, std::vector<std::int64_t>& unknown_indices)
{
  const int mpi_size = dolfinx::MPI::size(comm);
  const std::int64_t global_space
      = dolfinx::MPI::max(comm, *std::max_element(unknown_indices.begin(),
                                                  unknown_indices.end()))
        + 1;

  std::vector<std::vector<std::int64_t>> send_indices(mpi_size);
  for (std::int64_t global_i : unknown_indices)
  {
    const int index_owner
        = dolfinx::MPI::index_owner(mpi_size, global_i, global_space);
    send_indices[index_owner].push_back(global_i);
  }

  std::vector<std::vector<std::int64_t>> recv_indices(mpi_size);
  dolfinx::MPI::all_to_all(comm, send_indices, recv_indices);

  // Get index sharing - first vector entry (lowest) is owner.
  std::unordered_map<std::int64_t, std::vector<int>> index_to_owner;
  for (int p = 0; p < mpi_size; ++p)
  {
    const std::vector<std::int64_t>& recv_p = recv_indices[p];
    for (std::size_t j = 0; j < recv_p.size(); ++j)
      index_to_owner[recv_p[j]].push_back(p);
  }

  // Send index ownership data back to all sharing processes
  std::vector<std::vector<int>> send_owner(mpi_size);
  for (int p = 0; p < mpi_size; ++p)
  {
    const std::vector<std::int64_t>& recv_p = recv_indices[p];
    for (std::size_t j = 0; j < recv_p.size(); ++j)
    {
      const auto it = index_to_owner.find(recv_p[j]);
      assert(it != index_to_owner.end());
      const std::vector<int>& sharing_procs = it->second;
      send_owner[p].push_back(sharing_procs.size());
      for (int sp : sharing_procs)
        send_owner[p].push_back(sp);
    }
  }

  std::vector<std::vector<int>> recv_owner(mpi_size);
  dolfinx::MPI::all_to_all(comm, send_owner, recv_owner);

  // Now fill index_to_owner with locally needed indices
  index_to_owner.clear();
  for (int p = 0; p < mpi_size; ++p)
  {
    const std::vector<std::int64_t>& send_v = send_indices[p];
    const std::vector<int>& r_owner = recv_owner[p];
    int c = 0;
    int i = 0;
    while (c < (int)r_owner.size())
    {
      int count = r_owner[c];
      ++c;
      for (int j = 0; j < count; ++j)
      {
        index_to_owner[send_v[i]].push_back(r_owner[c]);
        ++c;
      }
      ++i;
    }
  }

  return index_to_owner;
}

} // namespace

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
Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>
mesh::compute_edge_reflections(const Topology& topology)
{
  const int tdim = topology.dim();
  const CellType cell_type = topology.cell_type();
  const int edges_per_cell = cell_num_entities(cell_type, 1);

  auto c_to_v = topology.connectivity(tdim, 0);
  assert(c_to_v);
  auto c_to_e = topology.connectivity(tdim, 1);
  assert(c_to_e);
  auto e_to_v = topology.connectivity(1, 0);
  assert(e_to_v);

  Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> reflections(
      edges_per_cell, c_to_v->num_nodes());
  for (int c = 0; c < c_to_v->num_nodes(); ++c)
  {
    auto cell_vertices = c_to_v->links(c);
    auto cell_edges = c_to_e->links(c);
    for (int i = 0; i < edges_per_cell; ++i)
    {
      auto vertices = e_to_v->links(cell_edges[i]);

      // If the entity is an interval, it should be oriented pointing
      // from the lowest numbered vertex to the highest numbered vertex.

      // Find iterators pointing to cell vertex given a vertex on facet
      const auto it0
          = std::find(cell_vertices.data(),
                      cell_vertices.data() + cell_vertices.size(), vertices[0]);
      const auto it1
          = std::find(cell_vertices.data(),
                      cell_vertices.data() + cell_vertices.size(), vertices[1]);

      // The number of reflections. Comparing iterators directly instead
      // of values they point to is sufficient here.
      reflections(i, c) = (it1 < it0);
    }
  }

  return reflections;
}
//-----------------------------------------------------------------------------
std::pair<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>,
          Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
mesh::compute_face_permutations(const Topology& topology)
{
  const int tdim = topology.dim();
  assert(tdim > 2);
  if (!topology.index_map(2))
    throw std::runtime_error("Faces have not been computed");

  // If faces have been computed, the below should exist
  auto c_to_v = topology.connectivity(tdim, 0);
  assert(c_to_v);
  auto c_to_f = topology.connectivity(tdim, 2);
  assert(c_to_f);
  auto f_to_v = topology.connectivity(2, 0);
  assert(f_to_v);

  const CellType cell_type = topology.cell_type();
  const int faces_per_cell = cell_num_entities(cell_type, 2);
  if (cell_type == CellType::triangle or cell_type == CellType::tetrahedron)
  {
    return compute_face_permutations_simplex(*c_to_v, *c_to_f, *f_to_v,
                                             faces_per_cell);
  }
  else
  {
    return compute_face_permutations_tp(*c_to_v, *c_to_f, *f_to_v,
                                        faces_per_cell);
  }
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
const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>&
Topology::get_edge_reflections() const
{
  return _edge_reflections;
}
//-----------------------------------------------------------------------------
const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>&
Topology::get_face_reflections() const
{
  return _face_reflections;
}
//-----------------------------------------------------------------------------
const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>&
Topology::get_face_rotations() const
{
  return _face_rotations;
}
//-----------------------------------------------------------------------------
const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>&
Topology::get_facet_permutations() const
{
  return _facet_permutations;
}
//-----------------------------------------------------------------------------
void Topology::create_entity_permutations()
{
  if (_edge_reflections.rows() > 0)
  {
    // assert(_face_reflections.size() != 0);
    // assert(_face_rotations.size() != 0);
    // assert(_facet_permutations.size() != 0);
    return;
  }

  const int tdim = this->dim();
  const CellType cell_type = this->cell_type();
  assert(_connectivity(tdim, 0));
  const std::int32_t num_cells = _connectivity(tdim, 0)->num_nodes();
  const int faces_per_cell = cell_num_entities(cell_type, 2);

  // FIXME: Avoid create 'identity' reflections/rotations

  if (tdim <= 1)
  {
    const int edges_per_cell = cell_num_entities(cell_type, 1);
    const int facets_per_cell = cell_num_entities(cell_type, tdim - 1);
    _edge_reflections.resize(edges_per_cell, num_cells);
    _edge_reflections = false;

    _face_reflections.resize(faces_per_cell, num_cells);
    _face_reflections = false;

    _face_rotations.resize(faces_per_cell, num_cells);
    _face_rotations = 0;
    _facet_permutations.resize(facets_per_cell, num_cells);
    _facet_permutations = 0;
  }

  if (tdim > 1)
  {
    _face_reflections.resize(faces_per_cell, num_cells);
    _face_reflections = false;
    _face_rotations.resize(faces_per_cell, num_cells);
    _face_rotations = 0;

    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> reflections
        = mesh::compute_edge_reflections(*this);
    _edge_reflections = reflections;
    if (tdim == 2)
      _facet_permutations = reflections.cast<std::uint8_t>();
  }

  if (tdim > 2)
  {
    auto [reflections, rotations] = mesh::compute_face_permutations(*this);
    _face_reflections = reflections;
    _face_rotations = rotations;
    if (tdim == 3)
      _facet_permutations = 2 * rotations + reflections.cast<std::uint8_t>();
  }
}
//-----------------------------------------------------------------------------
mesh::CellType Topology::cell_type() const { return _cell_type; }
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
std::tuple<Topology, std::vector<int>, graph::AdjacencyList<std::int32_t>>
mesh::create_topology(MPI_Comm comm,
                      const graph::AdjacencyList<std::int64_t>& cells,
                      const fem::ElementDofLayout& layout,
                      mesh::GhostMode ghost_mode)
{
  const int size = dolfinx::MPI::size(comm);

  // TODO: This step can be skipped for 'P1' elements

  // Extract topology data, e.g.just the vertices. For P1 geometry this
  // should just be the identity operator.For other elements the
  // filtered lists may have 'gaps', i.e.the indices might not be
  // contiguous.
  const graph::AdjacencyList<std::int64_t> cells_v
      = mesh::extract_topology(layout, cells);

  // Compute the destination rank for cells on this process via graph
  // partitioning
  const graph::AdjacencyList<std::int32_t> dest = Partitioning::partition_cells(
      comm, size, layout.cell_type(), cells_v, ghost_mode);

  // Distribute cells to destination rank
  const auto [my_cells, src, original_cell_index, ghost_owners]
      = graph::Partitioning::distribute(comm, cells_v, dest);

  // Get indices of ghost cells, if any
  const std::vector<std::int64_t> ghost_indices
      = graph::Partitioning::compute_ghost_indices(comm, original_cell_index,
                                                   ghost_owners);

  // Build local cell-vertex connectivity, with local vertex indices
  // [0, 1, 2, ..., n), from cell-vertex connectivity using global
  // indices and get map from global vertex indices in 'cells' to the
  // local vertex indices
  auto [cells_local, local_to_global_vertices]
      = graph::Partitioning::create_local_adjacency_list(my_cells);

  // Cell IndexMap
  const int num_local_cells = cells_local.num_nodes() - ghost_indices.size();
  auto index_map_c = std::make_shared<common::IndexMap>(comm, num_local_cells,
                                                        ghost_indices, 1);

  if (ghost_indices.size() > 0)
  {
    std::cout << "Ghosted mesh...\n";

    // Map from existing global vertex index to a local index
    // putting unowned indices last
    std::unordered_map<std::int64_t, std::int32_t> global_to_local_index;

    // Any vertices which are in ghost cells set to -1 since we need to
    // determine ownership
    for (std::size_t i = 0; i < ghost_indices.size(); ++i)
    {
      auto v = my_cells.links(num_local_cells + i);
      for (int j = 0; j < v.size(); ++j)
      {
        std::int64_t gi = v[j];
        global_to_local_index.insert({gi, -1});
      }
    }

    // Make a list of all vertex indices whose ownership needs determining
    std::vector<std::int64_t> unknown_indices;
    for (auto q : global_to_local_index)
      unknown_indices.push_back(q.first);

    std::unordered_map<std::int64_t, std::vector<int>> global_to_procs
        = compute_sharing(comm, unknown_indices);

    // Get all global vertex indices
    // FIXME: unordered may be better?
    std::set<std::int64_t> global_vertex_set(my_cells.array().data(),
                                             my_cells.array().data()
                                                 + my_cells.array().size());

    // Number all indices which this process now owns
    int mpi_rank = MPI::rank(comm);
    std::int32_t c = 0;
    for (std::int64_t gi : global_vertex_set)
    {
      const auto it = global_to_procs.find(gi);
      if (it != global_to_procs.end())
      {
        // Shared and locally owned
        if (it->second[0] == mpi_rank)
        {
          // Should already be in map, but needs index
          auto it_gi = global_to_local_index.find(gi);
          assert(it_gi != global_to_local_index.end());
          it_gi->second = c;
          ++c;
        }
      }
      else
      {
        // Unshared, and locally owned
        auto [it_ignore, insert] = global_to_local_index.insert({gi, c});
        assert(insert);
        ++c;
      }
    }
    std::int32_t nlocal = c;

    // Get global offset for local indices
    std::int64_t global_offset
        = dolfinx::MPI::global_offset(comm, nlocal, true);

    // Number ghosts locally
    for (auto& q : global_to_local_index)
    {
      if (q.second == -1)
      {
        q.second = c;
        ++c;
      }
    }
    std::int32_t nghosts = c - nlocal;

    // Convert my_cells (global indexing) to my_local_cells (local indexing)
    const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& my_cells_array
        = my_cells.array();
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> my_local_cells_array(
        my_cells_array.size());
    for (int i = 0; i < my_local_cells_array.size(); ++i)
      my_local_cells_array[i] = global_to_local_index[my_cells_array[i]];
    auto my_local_cells = std::make_shared<graph::AdjacencyList<std::int32_t>>(
        my_local_cells_array, my_cells.offsets());

    // Find all vertex-sharing neighbours, and process-to-neighbour map
    std::set<int> vertex_neighbours;
    for (const auto q : global_to_procs)
      vertex_neighbours.insert(q.second.begin(), q.second.end());
    vertex_neighbours.erase(mpi_rank);
    std::vector<int> neighbours(vertex_neighbours.begin(),
                                vertex_neighbours.end());
    std::unordered_map<int, int> proc_to_neighbours;
    for (std::size_t i = 0; i < neighbours.size(); ++i)
      proc_to_neighbours.insert({neighbours[i], i});

    // Communicate new global index to neighbours
    MPI_Comm neighbour_comm;
    MPI_Dist_graph_create_adjacent(comm, neighbours.size(), neighbours.data(),
                                   MPI_UNWEIGHTED, neighbours.size(),
                                   neighbours.data(), MPI_UNWEIGHTED,
                                   MPI_INFO_NULL, false, &neighbour_comm);

    // Get count and offset for each neighbour
    std::vector<int> send_counts(neighbours.size());
    for (auto q : global_to_procs)
    {
      const std::vector<int>& procs = q.second;
      if (procs[0] == mpi_rank)
      {
        for (std::size_t j = 1; j < procs.size(); ++j)
        {
          const int p = procs[j];
          const int np = proc_to_neighbours[p];
          send_counts[np] += 2;
        }
      }
    }
    std::vector<int> send_offsets = {0};
    for (int c : send_counts)
      send_offsets.push_back(send_offsets.back() + c);
    std::vector<int> temp_offsets(send_offsets.begin(), send_offsets.end());
    std::vector<std::int64_t> send_data(send_offsets.back());

    for (auto q : global_to_procs)
    {
      const std::vector<int>& procs = q.second;
      if (procs[0] == mpi_rank)
      {
        auto it = global_to_local_index.find(q.first);
        assert(it != global_to_local_index.end());

        // Owned and shared with these processes
        for (std::size_t j = 1; j < procs.size(); ++j)
        {
          int np = proc_to_neighbours[procs[j]];
          int& pos = temp_offsets[np];
          send_data[pos] = it->first;
          ++pos;
          send_data[pos] = it->second + global_offset;
          ++pos;
        }
      }
    }

    for (std::size_t i = 0; i < neighbours.size(); ++i)
      assert(send_offsets[i + 1] == temp_offsets[i]);

    std::vector<std::int64_t> recv_data;
    std::vector<int> recv_offsets;
    dolfinx::MPI::neighbor_all_to_all(neighbour_comm, send_offsets, send_data,
                                      recv_offsets, recv_data);
    MPI_Comm_free(&neighbour_comm);

    std::vector<std::int64_t> ghost_vertices(nghosts, -1);
    // Unpack received data and make list of ghosts
    for (std::size_t i = 0; i < recv_data.size(); i += 2)
    {
      std::int64_t gi = recv_data[i];
      const auto it = global_to_local_index.find(gi);
      assert(it != global_to_local_index.end());
      ghost_vertices[it->second - nlocal] = recv_data[i + 1];
    }
    // Check all ghosts are filled
    for (std::int64_t v : ghost_vertices)
      assert(v != -1);

    Topology topology(layout.cell_type());
    const int tdim = topology.dim();

    // Vertex IndexMap
    auto index_map_v
        = std::make_shared<common::IndexMap>(comm, nlocal, ghost_vertices, 1);
    topology.set_index_map(0, index_map_v);
    auto c0 = std::make_shared<graph::AdjacencyList<std::int32_t>>(
        index_map_v->size_local() + index_map_v->num_ghosts());
    topology.set_connectivity(c0, 0, 0);

    // Cell IndexMap
    topology.set_index_map(tdim, index_map_c);
    topology.set_connectivity(my_local_cells, tdim, 0);

    return {topology, src, dest};
  }

  // Create (i) local topology object and (ii) IndexMap for cells, and
  // set cell-vertex topology
  Topology topology_local(layout.cell_type());
  const int tdim = topology_local.dim();
  topology_local.set_index_map(tdim, index_map_c);

  auto _cells_local
      = std::make_shared<graph::AdjacencyList<std::int32_t>>(cells_local);
  topology_local.set_connectivity(_cells_local, tdim, 0);

  const int n = local_to_global_vertices.size();
  auto map = std::make_shared<common::IndexMap>(comm, n,
                                                std::vector<std::int64_t>(), 1);
  topology_local.set_index_map(0, map);
  auto _vertices_local
      = std::make_shared<graph::AdjacencyList<std::int32_t>>(n);
  topology_local.set_connectivity(_vertices_local, 0, 0);

  // Create facets for local topology, and attach to the topology
  // object. This will be used to find possibly shared cells
  auto [cf, fv, map0]
      = TopologyComputation::compute_entities(comm, topology_local, tdim - 1);
  if (cf)
    topology_local.set_connectivity(cf, tdim, tdim - 1);
  if (map0)
    topology_local.set_index_map(tdim - 1, map0);
  if (fv)
    topology_local.set_connectivity(fv, tdim - 1, 0);
  auto [fc, ignore] = TopologyComputation::compute_connectivity(topology_local,
                                                                tdim - 1, tdim);
  if (fc)
    topology_local.set_connectivity(fc, tdim - 1, tdim);

  // FIXME: This looks weird. Revise.
  // Get facets that are on the boundary of the local topology, i.e
  // are connect to one cell only
  std::vector<bool> boundary = compute_interior_facets(topology_local);
  topology_local.set_interior_facets(boundary);
  boundary = topology_local.on_boundary(tdim - 1);

  // Build distributed cell-vertex AdjacencyList, IndexMap for
  // vertices, and map from local index to old global index
  const std::vector<bool>& exterior_vertices
      = Partitioning::compute_vertex_exterior_markers(topology_local);
  auto [cells_d, vertex_map]
      = graph::Partitioning::create_distributed_adjacency_list(
          comm, *_cells_local, local_to_global_vertices, exterior_vertices);

  Topology topology(layout.cell_type());

  // Set vertex IndexMap, and vertex-vertex connectivity
  auto _vertex_map = std::make_shared<common::IndexMap>(std::move(vertex_map));
  topology.set_index_map(0, _vertex_map);
  auto c0 = std::make_shared<graph::AdjacencyList<std::int32_t>>(
      _vertex_map->size_local() + _vertex_map->num_ghosts());
  topology.set_connectivity(c0, 0, 0);

  // Set cell IndexMap and cell-vertex connectivity
  //  auto index_map_c = std::make_shared<common::IndexMap>(
  //      comm, cells_d.num_nodes(), std::vector<std::int64_t>(), 1);
  topology.set_index_map(tdim, index_map_c);
  auto _cells_d = std::make_shared<graph::AdjacencyList<std::int32_t>>(cells_d);
  topology.set_connectivity(_cells_d, tdim, 0);

  return {topology, src, dest};
}
//-----------------------------------------------------------------------------
