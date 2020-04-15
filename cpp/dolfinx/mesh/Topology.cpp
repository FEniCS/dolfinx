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
// Given a list of indices (unknown_indices) on each process, return a
// map to sharing processes for each index, taking the owner as the
// first in the list
std::unordered_map<std::int64_t, std::vector<int>>
compute_index_sharing(MPI_Comm comm, std::vector<std::int64_t>& unknown_indices)
{
  const int mpi_size = dolfinx::MPI::size(comm);
  std::int64_t global_space = 0;
  const std::int64_t max_index
      = *std::max_element(unknown_indices.begin(), unknown_indices.end());
  MPI_Allreduce(&max_index, &global_space, 1, MPI_INT64_T, MPI_SUM, comm);
  global_space += 1;

  std::vector<std::vector<std::int64_t>> send_indices(mpi_size);
  for (std::int64_t global_i : unknown_indices)
  {
    const int index_owner
        = dolfinx::MPI::index_owner(mpi_size, global_i, global_space);
    send_indices[index_owner].push_back(global_i);
  }

  const graph::AdjacencyList<std::int64_t> recv_indices
      = dolfinx::MPI::all_to_all(
          comm, graph::AdjacencyList<std::int64_t>(send_indices));

  // Get index sharing - first vector entry (lowest) is owner.
  std::unordered_map<std::int64_t, std::vector<int>> index_to_owner;
  for (int p = 0; p < recv_indices.num_nodes(); ++p)
  {
    auto recv_p = recv_indices.links(p);
    for (int j = 0; j < recv_p.rows(); ++j)
      index_to_owner[recv_p[j]].push_back(p);
  }

  // Send index ownership data back to all sharing processes
  std::vector<std::vector<int>> send_owner(mpi_size);
  for (int p = 0; p < recv_indices.num_nodes(); ++p)
  {
    auto recv_p = recv_indices.links(p);
    for (int j = 0; j < recv_p.rows(); ++j)
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
  const graph::AdjacencyList<int> recv_owner
      = dolfinx::MPI::all_to_all(comm, graph::AdjacencyList<int>(send_owner));

  // Now fill index_to_owner with locally needed indices
  index_to_owner.clear();
  for (int p = 0; p < mpi_size; ++p)
  {
    const std::vector<std::int64_t>& send_v = send_indices[p];
    auto r_owner = recv_owner.links(p);
    int c(0), i(0);
    while (c < r_owner.rows())
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
Eigen::Array<std::int64_t, Eigen::Dynamic, 1>
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

  return dolfinx::MPI::neighbor_all_to_all(neighbour_comm, qsend_offsets,
                                           qsend_data)
      .array();
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
int Topology::dim() const { return cell_dim(cell_type()); }
//-----------------------------------------------------------------------------
std::shared_ptr<const common::IndexMap>
Topology::index_map(int dim, bool discard_intermediate /*unused*/) const
{
  // "discard_intermediate" is not used here because the "intermediate results"
  // are logically connect to the index map and thus not truly "intermediate".

  if (auto res = cache.index_map(dim); res)
  {
    // They usually belong together.
    // There may be issues with TopologyComputation::compute_entities and at
    // various other places.
    assert(cache.connectivity(this->dim(), dim));
    assert(cache.connectivity(dim, 0));
    return res;
  }

  // General lock
  auto lock = acquire_cache_lock();
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> _entity_vertex;
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> _cell_entity;
  std::shared_ptr<const common::IndexMap> _index_map;

  // Create local entities
  std::tie(_cell_entity, _entity_vertex, _index_map)
  = TopologyComputation::compute_entities(*this, dim);

  cache.set_connectivity(_cell_entity, this->dim(), dim);
  cache.set_connectivity(_entity_vertex, dim, 0);
  return cache.set_index_map(_index_map, dim);
}
//-----------------------------------------------------------------------------
std::vector<bool> Topology::on_boundary(int dim,
                                        bool discard_intermediate) const
{
  const int tdim = this->dim();
  if (dim >= tdim or dim < 0)
  {
    throw std::runtime_error("Invalid entity dimension: "
                             + std::to_string(dim));
  }

  std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
      connectivity_facet_cell
      = connectivity(tdim - 1, tdim, discard_intermediate);

  // TODO: figure out if we can/should make this for owned entities only
  auto _index_map = index_map(dim);
  std::vector<bool> marker(
      index_map(dim)->size_local() + index_map(dim)->num_ghosts(), false);
  const int num_facets
      = index_map(dim - 1)->size_local() + index_map(dim - 1)->num_ghosts();

  // Special case for facets
  if (dim == tdim - 1)
  {
    auto facets = interior_facets(discard_intermediate);
    assert(num_facets <= static_cast<int>(facets.size()));
    assert(num_facets <= static_cast<int>(marker.size()));
    std::transform(begin(facets), begin(facets) + num_facets, begin(marker),
                   [](const bool& interior_facet) { return !interior_facet; });
    return marker;
  }

  // Get connectivity from facet to entities of interest (vertices or edges)
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
      connectivity_facet_entity
      = connectivity(tdim - 1, dim, discard_intermediate);
  if (!connectivity_facet_entity)
    throw std::runtime_error("Facet-entity connectivity missing");

  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& fe_offsets
      = connectivity_facet_entity->offsets();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& fe_indices
      = connectivity_facet_entity->array();

  // Iterate over all facets, selecting only those with one cell
  // attached
  auto facets = interior_facets(discard_intermediate);
  assert(num_facets <= static_cast<int>(facets.size()));

  for (int i = 0; i < num_facets; ++i)
  {
    // True if facet is on boundary (= not interior)
    if (!facets[i])
    {
      for (int j = fe_offsets[i]; j < fe_offsets[i + 1]; ++j)
        marker[fe_indices[j]] = true;
    }
  }

  return marker;
}
//------------------------------------------------------------------------------
std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
Topology::connectivity(int d0, int d1, bool discard_intermediate) const
{
  if (auto conn = cache.connectivity(d0, d1); conn)
    return conn;

  // connected index maps:
  //  im (dim)         |       topologies (d0, d1)
  //    x                      tdim, x  &  x, 0
  //  dims 0 and tdim are always valid by construction
  //  => Attention for
  //   * (y, x) with y = tdim and
  //   * (x, z) with z = 0
  // In such cases, the topology (a) must not be cache stored or (b) everything
  // by create entities (x) must be cached (more expensive).
  // Use option discard_intermediate to distinguish.
  //
  // NOTE: TopologyComputation::compute_topology will throw in such cases!
  //
  // Using the same checks here. Not all four cases are checked since they are
  // not logically independent.

  // General lock to ensure a writable cache layer
  auto lock = acquire_cache_lock();
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> res;
  // Special cases for which we cannot discard intermediate data.
  if (d0 == dim() or d1 == 0)
  {
    // This calls do not do excessive work since they are either computed
    // or essential, i.e. present by construction.
    // case (d0, 0): computes connectivities (tdim, d0) and (d0, 0)
    index_map(d0);
    // case (tdim, d1) computes connectivities (tdim, d1) and (d1, 0)
    index_map(d1);
    assert(cache.connectivity(d0, d1));
    return cache.connectivity(d0, d1);
  }
  else
  {
    { // scope for temporary writing cache
      auto lock = acquire_cache_lock(discard_intermediate);
      // index_map(dim) triggers create_entities.
      if (!cache.connectivity(d0, 0))
        index_map(d0);

      if (!cache.connectivity(d1, 0))
        index_map(d1);

      res = TopologyComputation::compute_connectivity(*this, d0, d1)[0];
    } // discard temporary cache

    // Store result because it may have been dropped
    return cache.set_connectivity(res, d0, d1);
  }
}
//-----------------------------------------------------------------------------
const std::vector<bool>&
Topology::interior_facets(bool discard_intermediate) const
{
  if (auto facets = cache.interior_facets(); facets)
    return *facets;

  // General lock to be operable in this function
  auto lock = acquire_cache_lock();
  std::shared_ptr<const std::vector<bool>> res;

  { // scope for optionally discarded cache layer
    auto lock = acquire_cache_lock(discard_intermediate);
    res = std::make_shared<const std::vector<bool>>(
        TopologyComputation::compute_interior_facets(*this));
  } // discard cache layer

  // Store result because it may have been dropped
  return *(cache.set_interior_facets(res));
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
Topology::get_cell_permutation_info(bool discard_intermediate) const
{
  // Note: discard intermediate does not apply to facet_permutations which
  // are computed as well.

  if (auto res = cache.cell_permutations(); res)
  {
    assert(cache.facet_permutations());
    return *res;
  }

  // General lock
  auto lock = acquire_cache_lock();
  std::shared_ptr<
      const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
      facet_permutations;
  std::shared_ptr<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
      cell_permutations;

  { // scope for optionally discarded cache layer
    auto discard_lock = acquire_cache_lock(discard_intermediate);
    // Requires entities: trigger via call to index_map
    for (int d = 0; d < dim(); ++d)
      index_map(d);

    auto [_facet_permutations, _cell_permutations]
        = PermutationComputation::compute_entity_permutations(*this);

    facet_permutations = std::make_shared<
        const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>(
        std::move(_facet_permutations));
    cell_permutations = std::make_shared<
        const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>(
        std::move(_cell_permutations));
  } // leave scope of discard_lock

  // Write since layer may have been discard
  cache.set_facet_permutations(facet_permutations);
  return *(cache.set_cell_permutations(cell_permutations));
}
//-----------------------------------------------------------------------------
const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>&
Topology::get_facet_permutations(bool discard_intermediate) const
{
  if (auto facet_permutations = cache.facet_permutations(); facet_permutations)
  {
    assert(cache.cell_permutations());
    return *facet_permutations;
  }

  // Rely on get_cell_permutation_info. The lock ensures transfer of cached data.
  auto lock = acquire_cache_lock(discard_intermediate);
  get_cell_permutation_info(discard_intermediate);
  return *(cache_data().facet_permutations());
}
//-----------------------------------------------------------------------------
mesh::CellType Topology::cell_type() const { return _cell_type; }
//-----------------------------------------------------------------------------
MPI_Comm Topology::mpi_comm() const { return _mpi_comm.comm(); }
//-----------------------------------------------------------------------------
storage::TopologyStorage
Topology::check_storage(storage::TopologyStorage remanent_storage, int tdim)
{
  if (!(remanent_storage.index_map(tdim) && remanent_storage.index_map(0)
        && remanent_storage.connectivity(tdim, 0)
        && remanent_storage.connectivity(0, 0)))
    throw std::invalid_argument("Storage does not provide all required data: "
                                "index_map(0), index_map(tdim), "
                                "connectivity(0, 0), connectivity(tdim, 0).");
  return std::move(remanent_storage);
}
//------------------------------------------------------------------------------
std::int32_t Topology::create_entities(int dim)
{
  if (auto conn
      = remanent_storage.set_connectivity(cache.connectivity(dim, 0), dim, 0);
      conn)
  {
    remanent_storage.set_index_map(cache.index_map(dim), dim);
    remanent_storage.set_connectivity(cache.connectivity(this->dim(), dim),
                                      this->dim(), dim);
    return -1;
  }

  // Create local entities
  const auto [cell_entity, entity_vertex, index_map]
      = TopologyComputation::compute_entities(*this, dim);

  // cell_entitity should not be empty because of the check above
  assert(cell_entity);
  remanent_storage.set_connectivity(cell_entity, this->dim(), dim);

  // entitiy_vertex should not be empty because of the check above
  assert(entity_vertex);
  remanent_storage.set_connectivity(entity_vertex, dim, 0);

  // index_map should not be empty because of the check above
  assert(index_map);
  remanent_storage.set_index_map(index_map, dim);

  return index_map->size_local();
}
//--------------------------------------------------------------------------
void Topology::create_connectivity(int d0, int d1)
{

  // Are the parens required?
  if (auto conn
      = remanent_storage.set_connectivity(cache.connectivity(d0, d1), d0, d1);
      conn)
  {
    // Probably it's better to also copy the shared_ptr to index maps
    remanent_storage.set_index_map(cache.index_map(d0), d0);
    remanent_storage.set_index_map(cache.index_map(d1), d1);
    return;
  }

  // TODO: this means , they will be stored! Not necessary since they will be
  // computed on demand otherwise Make sure entities exist
  create_entities(d0);
  create_entities(d1);

  // Compute connectivity
  const auto [c_d0_d1, c_d1_d0]
      = TopologyComputation::compute_connectivity(*this, d0, d1);

  // NOTE: that to compute the (d0, d1) connections is it sometimes
  // necessary to compute the (d1, d0) connections. We store the (d1,
  // d0) for possible later use, but there is a memory overhead if they
  // are not required. It may be better to not automatically store
  // connectivity that was not requested, but advise in a docstring the
  // most efficient order in which to call this function if several
  // connectivities are needed.

  // Concerning the note above: Provide an overload
  // create_connectivity(std::vector<std::pair<int, int>>)?

  // Attach connectivities
  if (c_d0_d1)
    remanent_storage.set_connectivity(c_d0_d1, d0, d1);
  if (c_d1_d0)
    remanent_storage.set_connectivity(c_d1_d0, d1, d0);

  // Special facet handing
  if (d0 == (this->dim() - 1) and d1 == this->dim())
  {
    // TODO: this means, they will be stored! Not necessary since they will be
    // computed on demand otherwise
    create_interior_facets();
  }
}
//-----------------------------------------------------------------------------
void Topology::create_connectivity_all()
{
  // Compute all connectivity
  for (int d0 = 0; d0 <= this->dim(); d0++)
    for (int d1 = 0; d1 <= this->dim(); d1++)
      create_connectivity(d0, d1);
}
//-----------------------------------------------------------------------------
void Topology::create_interior_facets()
{
  if (auto facets
      = remanent_storage.set_interior_facets(cache.interior_facets());
      facets)
    return;

  auto f = std::make_shared<std::vector<bool>>(
      TopologyComputation::compute_interior_facets(*this));
  remanent_storage.set_interior_facets(f);
}
//-----------------------------------------------------------------------------
void Topology::create_entity_permutations()
{
  if (auto permutations
      = remanent_storage.set_cell_permutations(cache.cell_permutations());
      permutations)
  {
    // Also copy facet permutations to storage
    remanent_storage.set_facet_permutations(cache.facet_permutations());
    return;
  }

  const int tdim = this->dim();

  // FIXME: Is this always required? Could it be made cheaper by doing a
  // local version? This call does quite a lot of parallel work
  // Create all mesh entities
  // TODO: drop the afterwards (via new layer of remanent storage or cache, when
  // not requiring them)?
  for (int d = 0; d < tdim; ++d)
    create_entities(d);

  auto [facet_permutations, cell_permutations]
      = PermutationComputation::compute_entity_permutations(*this);

  remanent_storage.set_facet_permutations(
      std::make_shared<
          const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>(
          std::move(facet_permutations)));

  remanent_storage.set_cell_permutations(
      std::make_shared<const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>(
          std::move(cell_permutations)));
}
//-----------------------------------------------------------------------------
Topology
mesh::create_topology(MPI_Comm comm,
                      const graph::AdjacencyList<std::int64_t>& cells,
                      const std::vector<std::int64_t>& original_cell_index,
                      const std::vector<int>& ghost_owners,
                      const CellType& cell_type, mesh::GhostMode)
{
  if (cells.num_nodes() > 0)
  {
    if (cells.num_links(0) != mesh::num_cell_vertices(cell_type))
    {
      throw std::runtime_error(
          "Inconsistent number of cell vertices. Got "
          + std::to_string(cells.num_links(0)) + ", expected "
          + std::to_string(mesh::num_cell_vertices(cell_type)) + ".");
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
    for (auto q : global_to_procs)
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
    for (const auto& q : global_to_procs)
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

    // FIXME: make const
    Eigen::Array<std::int64_t, Eigen::Dynamic, 1> recv_pairs
        = send_to_neighbours(neighbour_comm, send_pairs);

    std::vector<std::int64_t> ghost_vertices(nghosts, -1);
    // Unpack received data and make list of ghosts
    for (int i = 0; i < recv_pairs.rows(); i += 2)
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
    for (const auto& q : fwd_shared_vertices)
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
    for (int i = 0; i < recv_pairs.rows(); i += 2)
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

    storage::TopologyStorage storage(true);
    const int tdim = cell_dim(cell_type);

    // Vertex IndexMap
    auto index_map_v
        = std::make_shared<common::IndexMap>(comm, nlocal, ghost_vertices, 1);
    storage.set_index_map(index_map_v, 0);
    auto c0 = std::make_shared<graph::AdjacencyList<std::int32_t>>(
        index_map_v->size_local() + index_map_v->num_ghosts());
    storage.set_connectivity(c0, 0, 0);

    // Cell IndexMap
    storage.set_index_map(index_map_c, tdim);
    storage.set_connectivity(my_local_cells, tdim, 0);
    return Topology(comm, cell_type, std::move(storage));
  }

  // Build local cell-vertex connectivity, with local vertex indices [0,
  // 1, 2, ..., n), from cell-vertex connectivity using global indices
  // and get map from global vertex indices in 'cells' to the local
  // vertex indices
  auto [cells_local, local_to_global_vertices]
      = graph::Partitioning::create_local_adjacency_list(cells);

  storage::TopologyStorage storage_local{true};
  const int tdim = cell_dim(cell_type);

  storage_local.set_index_map(index_map_c, tdim);

  auto _cells_local
      = std::make_shared<graph::AdjacencyList<std::int32_t>>(cells_local);
  storage_local.set_connectivity(_cells_local, tdim, 0);

  const int n = local_to_global_vertices.size();
  auto map = std::make_shared<common::IndexMap>(comm, n,
                                                std::vector<std::int64_t>(), 1);
  storage_local.set_index_map(map, 0);
  auto _vertices_local
      = std::make_shared<graph::AdjacencyList<std::int32_t>>(n);
  storage_local.set_connectivity(_vertices_local, 0, 0);

  Topology topology_local(comm, cell_type, std::move(storage_local));

  // Create facets for local topology, and attach to the topology
  // object. This will be used to find possibly shared cells
  topology_local.create_connectivity(tdim, tdim - 1);
  topology_local.create_connectivity(tdim - 1, 0);
  topology_local.create_connectivity(tdim - 1, tdim);

  // Build distributed cell-vertex AdjacencyList, IndexMap for vertices,
  // and map from local index to old global index
  const std::vector<bool>& exterior_vertices
      = Partitioning::compute_vertex_exterior_markers(topology_local);
  auto [cells_d, vertex_map]
      = graph::Partitioning::create_distributed_adjacency_list(
          comm, *_cells_local, local_to_global_vertices, exterior_vertices);

  storage::TopologyStorage storage(true);

  // Set vertex IndexMap, and vertex-vertex connectivity
  auto _vertex_map = std::make_shared<common::IndexMap>(std::move(vertex_map));
  storage.set_index_map(_vertex_map, 0);
  auto c0 = std::make_shared<graph::AdjacencyList<std::int32_t>>(
      _vertex_map->size_local() + _vertex_map->num_ghosts());
  storage.set_connectivity(c0, 0, 0);

  // Set cell IndexMap and cell-vertex connectivity
  storage.set_index_map(index_map_c, tdim);
  auto _cells_d = std::make_shared<graph::AdjacencyList<std::int32_t>>(cells_d);
  storage.set_connectivity(_cells_d, tdim, 0);
  return Topology(comm, cell_type, std::move(storage));
}