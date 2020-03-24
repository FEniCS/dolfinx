// Copyright (C) 2006-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Topology.h"

#include "Partitioning.h"
#include "PermutationComputation.h"
#include "TopologyComputation.h"
#include "utils.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/Partitioning.h>
#include <numeric>
#include <unordered_map>

using namespace dolfinx;
using namespace dolfinx::mesh;

namespace
{
//-----------------------------------------------------------------------------
// Given a list of indices (unknown_indices) on each process,
// return a map to sharing processes for each index, taking the owner as the
// first in the list
std::unordered_map<std::int64_t, std::vector<int>>
compute_index_sharing(MPI_Comm comm, std::vector<std::int64_t>& unknown_indices)
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

  // Alltoall is necessary because cells which are shared by vertex are not yet
  // known to this process
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
      int count = r_owner[c++];
      for (int j = 0; j < count; ++j)
        index_to_owner[send_v[i]].push_back(r_owner[c++]);
      ++i;
    }
  }

  return index_to_owner;
}
//----------------------------------------------------------------------------------------------
// Wrapper around neighbor_all_to_all for vector<vector> style input
std::vector<std::int64_t>
send_to_neighbours(MPI_Comm neighbour_comm,
                   const std::vector<std::vector<std::int64_t>>& send_data)
{
  std::vector<int> qsend_offsets = {0};
  std::vector<std::int64_t> qsend_data;
  for (const std::vector<std::int64_t>& q : send_data)
  {
    qsend_data.insert(qsend_data.end(), q.begin(), q.end());
    qsend_offsets.push_back(qsend_data.size());
  }

  std::vector<std::int64_t> qrecv_data;
  std::vector<int> qrecv_offsets;
  dolfinx::MPI::neighbor_all_to_all(neighbour_comm, qsend_offsets, qsend_data,
                                    qrecv_offsets, qrecv_data);
  return qrecv_data;
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
Topology::Topology(mesh::CellType type)
    : _cell_type(type),
      _connectivity(mesh::cell_dim(type) + 1, mesh::cell_dim(type) + 1)
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
const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>&
Topology::get_cell_permutation_info() const
{
  return _cell_permutations;
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
  if (_cell_permutations.size() > 0)
    return;

  auto [facet_permutations, cell_permutations]
      = PermutationComputation::compute_entity_permutations(*this);
  _facet_permutations = std::move(facet_permutations);
  _cell_permutations = std::move(cell_permutations);
}
//-----------------------------------------------------------------------------
mesh::CellType Topology::cell_type() const { return _cell_type; }
//-----------------------------------------------------------------------------
Topology
mesh::create_topology(MPI_Comm comm,
                      const graph::AdjacencyList<std::int64_t>& cells,
                      const std::vector<std::int64_t>& original_cell_index,
                      const std::vector<int>& ghost_owners,
                      const fem::ElementDofLayout& layout, mesh::GhostMode)
{
  if (cells.num_nodes() > 0)
  {
    if (cells.num_links(0) != mesh::num_cell_vertices(layout.cell_type()))
    {
      throw std::runtime_error(
          "Inconsistent number of cell vertices. Got "
          + std::to_string(cells.num_links(0)) + ", expected "
          + std::to_string(mesh::num_cell_vertices(layout.cell_type())) + ".");
    }
  }

  // Get indices of ghost cells, if any
  const std::vector<std::int64_t> cell_ghost_indices
      = graph::Partitioning::compute_ghost_indices(comm, original_cell_index,
                                                   ghost_owners);

  // Cell IndexMap
  const int num_local_cells = cells.num_nodes() - cell_ghost_indices.size();
  auto index_map_c = std::make_shared<common::IndexMap>(comm, num_local_cells,
                                                        cell_ghost_indices, 1);

  if (cell_ghost_indices.size() > 0)
  {
    // Map from existing global vertex index to local index
    // putting ghost indices last
    std::unordered_map<std::int64_t, std::int32_t> global_to_local_index;

    // Any vertices which are in ghost cells set to -1 since we need to
    // determine ownership
    for (std::size_t i = 0; i < cell_ghost_indices.size(); ++i)
    {
      auto v = cells.links(num_local_cells + i);
      for (int j = 0; j < v.size(); ++j)
        global_to_local_index.insert({v[j], -1});
    }

    // Get all vertices which appear in both ghost and non-ghost cells
    // FIXME: optimize
    std::set<std::int64_t> ghost_boundary_vertices;
    std::set<std::int64_t> local_vertex_set;
    for (int i = 0; i < num_local_cells; ++i)
    {
      auto v = cells.links(i);
      for (int j = 0; j < v.size(); ++j)
      {
        auto it = global_to_local_index.find(v[j]);
        if (it != global_to_local_index.end())
          ghost_boundary_vertices.insert(v[j]);
        else
          local_vertex_set.insert(v[j]);
      }
    }

    int mpi_rank = MPI::rank(comm);

    // Make a list of all vertex indices whose ownership needs determining
    std::vector<std::int64_t> unknown_indices(ghost_boundary_vertices.begin(),
                                              ghost_boundary_vertices.end());
    std::unordered_map<std::int64_t, std::vector<int>> global_to_procs
        = compute_index_sharing(comm, unknown_indices);

    // Number all indices which this process now owns
    std::int32_t c = 0;
    for (std::int64_t gi : local_vertex_set)
    {
      // Locally owned
      auto [it_ignore, insert] = global_to_local_index.insert({gi, c++});
      assert(insert);
    }
    for (std::int64_t gi : ghost_boundary_vertices)
    {
      const auto it = global_to_procs.find(gi);
      assert(it != global_to_procs.end());

      // Shared and locally owned
      if (it->second[0] == mpi_rank)
      {
        // Should already be in map, but needs index
        auto it_gi = global_to_local_index.find(gi);
        assert(it_gi != global_to_local_index.end());
        assert(it_gi->second == -1);
        it_gi->second = c++;
      }
    }
    std::int32_t nlocal = c;
    std::int32_t nghosts = global_to_local_index.size() - nlocal;

    // Get global offset for local indices
    std::int64_t global_offset
        = dolfinx::MPI::global_offset(comm, nlocal, true);

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

    std::vector<std::vector<std::int64_t>> send_pairs(neighbours.size());
    for (auto q : global_to_procs)
    {
      const std::vector<int>& procs = q.second;
      if (procs[0] == mpi_rank)
      {
        auto it = global_to_local_index.find(q.first);
        assert(it != global_to_local_index.end());
        assert(it->second != -1);

        // Owned and shared with these processes
        // NB starting from 1. 0 is self.
        for (std::size_t j = 1; j < procs.size(); ++j)
        {
          int np = proc_to_neighbours[procs[j]];
          send_pairs[np].push_back(it->first);
          send_pairs[np].push_back(it->second + global_offset);
        }
      }
    }
    std::vector<std::int64_t> recv_pairs
        = send_to_neighbours(neighbour_comm, send_pairs);

    std::vector<std::int64_t> ghost_vertices(nghosts, -1);
    // Unpack received data and make list of ghosts
    for (std::size_t i = 0; i < recv_pairs.size(); i += 2)
    {
      std::int64_t gi = recv_pairs[i];
      const auto it = global_to_local_index.find(gi);
      assert(it != global_to_local_index.end());
      assert(it->second == -1);
      it->second = c++;
      ghost_vertices[it->second - nlocal] = recv_pairs[i + 1];
    }

    // At this point, this process should have indexed all "local" vertices.
    // Send out to processes which share them.
    std::map<std::int32_t, std::set<std::int32_t>> shared_cells
        = index_map_c->compute_shared_indices();
    std::map<std::int64_t, std::set<std::int32_t>> fwd_shared_vertices;
    for (int i = 0; i < index_map_c->size_local(); ++i)
    {
      auto it = shared_cells.find(i);
      if (it != shared_cells.end())
      {
        auto v = cells.links(i);
        for (int j = 0; j < v.size(); ++j)
        {
          auto vit = fwd_shared_vertices.find(v[j]);
          if (vit == fwd_shared_vertices.end())
            fwd_shared_vertices.insert({v[j], it->second});
          else
            vit->second.insert(it->second.begin(), it->second.end());
        }
      }
    }
    send_pairs = std::vector<std::vector<std::int64_t>>(neighbours.size());
    for (auto q : fwd_shared_vertices)
    {
      auto it = global_to_local_index.find(q.first);
      assert(it != global_to_local_index.end());
      assert(it->second != -1);

      std::int64_t gi;
      if (it->second < nlocal)
        gi = it->second + global_offset;
      else
        gi = ghost_vertices[it->second - nlocal];

      for (int p : q.second)
      {
        const int np = proc_to_neighbours[p];
        send_pairs[np].push_back(q.first);
        send_pairs[np].push_back(gi);
      }
    }
    recv_pairs = send_to_neighbours(neighbour_comm, send_pairs);

    // Unpack received data and add to ghosts
    for (std::size_t i = 0; i < recv_pairs.size(); i += 2)
    {
      std::int64_t gi = recv_pairs[i];
      const auto it = global_to_local_index.find(gi);
      assert(it != global_to_local_index.end());
      if (it->second == -1)
      {
        it->second = c++;
        ghost_vertices[it->second - nlocal] = recv_pairs[i + 1];
      }
    }
    MPI_Comm_free(&neighbour_comm);

    // Check all ghosts are filled
    assert(c = (nghosts + nlocal));
    for (std::int64_t v : ghost_vertices)
      assert(v != -1);

    // Convert my_cells (global indexing) to my_local_cells (local indexing)
    const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& cells_array
        = cells.array();
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> my_local_cells_array(
        cells_array.size());
    for (int i = 0; i < my_local_cells_array.size(); ++i)
      my_local_cells_array[i] = global_to_local_index[cells_array[i]];
    auto my_local_cells = std::make_shared<graph::AdjacencyList<std::int32_t>>(
        my_local_cells_array, cells.offsets());

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

    return topology;
  }

  // Build local cell-vertex connectivity, with local vertex indices
  // [0, 1, 2, ..., n), from cell-vertex connectivity using global
  // indices and get map from global vertex indices in 'cells' to the
  // local vertex indices
  auto [cells_local, local_to_global_vertices]
      = graph::Partitioning::create_local_adjacency_list(cells);

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
  topology.set_index_map(tdim, index_map_c);
  auto _cells_d = std::make_shared<graph::AdjacencyList<std::int32_t>>(cells_d);
  topology.set_connectivity(_cells_d, tdim, 0);

  return topology;
}
//-----------------------------------------------------------------------------
