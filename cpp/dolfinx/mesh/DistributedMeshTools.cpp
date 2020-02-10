// Copyright (C) 2011-2019 Garth N. Wells and Chris Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DistributedMeshTools.h"
#include "MeshFunction.h"
#include "cell_types.h"
#include "dolfinx/common/IndexMap.h"
#include "dolfinx/common/MPI.h"
#include "dolfinx/common/Timer.h"
#include "dolfinx/graph/Graph.h"
#include "dolfinx/graph/SCOTCH.h"
#include <Eigen/Dense>
#include <complex>
#include <dolfinx/common/log.h>

using namespace dolfinx;
using namespace dolfinx::mesh;

//-----------------------------------------------------------------------------
namespace
{
std::tuple<std::vector<std::int64_t>,
           std::map<std::int32_t, std::set<std::int32_t>>,
           std::shared_ptr<const common::IndexMap>>
compute_entity_numbering(MPI_Comm comm, const Topology& topology,
                         const mesh::CellType cell_type, int d)
{
  LOG(INFO) << "Number mesh entities for distributed mesh. " << d;
  common::Timer timer("Number mesh entities for distributed mesh");

  std::vector<std::int64_t> global_entity_indices;
  std::map<std::int32_t, std::set<std::int32_t>> shared_entities;

  // Check that we're not re-numbering vertices (these are fixed at mesh
  // construction)
  if (d == 0 or d == topology.dim())
  {
    throw std::runtime_error(
        "Global vertex and cell indices exist at input. Cannot be renumbered.");
  }

  // Get number of processes and process number
  const int mpi_size = dolfinx::MPI::size(comm);
  const int mpi_rank = dolfinx::MPI::rank(comm);

  // Get vertex global indices
  const std::vector<std::int64_t>& global_vertex_indices
      = topology.global_indices(0);

  // Get shared vertices (local index, [sharing processes])
  // already determined in Mesh distribution
  const std::map<std::int32_t, std::set<std::int32_t>>& shared_vertices_local
      = topology.shared_entities(0);

  // Get number of entities of dimension d on this process
  auto c_d_0 = topology.connectivity(d, 0);
  assert(c_d_0);
  const std::int32_t size = c_d_0->num_nodes();

  // Send and receive shared entities.
  // In order to communicate with remote processes, the entities are sent
  // and received as blocks of global vertex indices, since these are the same
  // on all processes. However, is more convenient to use the local index, so
  // translate before and after sending.
  std::vector<std::vector<std::int32_t>> send_local_entities(mpi_size);
  std::vector<std::vector<std::int32_t>> recv_local_entities(mpi_size);

  {
    // Scoped region

    // Entities to send/recv, as blocks of global vertex indices.
    std::vector<std::vector<std::int64_t>> send_entities(mpi_size);
    std::vector<std::vector<std::int64_t>> recv_entities(mpi_size);

    // Mapping from the global vertex indices of an entity, back to its local
    // index. Used after receiving, to translate back to local index.
    std::map<std::vector<std::int64_t>, std::int32_t> entity_to_local_index;

    // Get number of vertices in this entity type
    const mesh::CellType& et = mesh::cell_entity_type(cell_type, d);
    const int num_entity_vertices = mesh::num_cell_vertices(et);

    // Make a mask listing all shared vertices (true if shared)
    auto map0 = topology.index_map(0);
    assert(map0);
    const int num_vertices = map0->size_local() + map0->num_ghosts();
    std::vector<bool> vertex_shared(num_vertices, false);
    for (const auto& v : shared_vertices_local)
      vertex_shared[v.first] = true;

    for (int e = 0; e < size; ++e)
    {
      auto v = c_d_0->links(e);

      // Entity can only be shared if all vertices are shared but this
      // is not a sufficient condition
      bool may_be_shared = true;
      for (int i = 0; i < num_entity_vertices; ++i)
        may_be_shared &= vertex_shared[v[i]];

      if (may_be_shared)
      {
        // Get the set of processes which share all the vertices of this entity

        // First vertex, and its sharing processes
        const std::set<std::int32_t>& shared_vertices_v0
            = shared_vertices_local.find(v[0])->second;
        std::vector<std::int32_t> sharing_procs(shared_vertices_v0.begin(),
                                                shared_vertices_v0.end());

        for (int i = 1; i < num_entity_vertices; ++i)
        {
          // Sharing processes for subsequent vertices of this entity
          const std::set<std::int32_t>& shared_vertices_v
              = shared_vertices_local.find(v[i])->second;

          std::vector<std::int32_t> v_sharing_procs;
          std::set_intersection(sharing_procs.begin(), sharing_procs.end(),
                                shared_vertices_v.begin(),
                                shared_vertices_v.end(),
                                std::back_inserter(v_sharing_procs));
          sharing_procs = v_sharing_procs;
        }

        if (!sharing_procs.empty())
        {
          // If there are still processes that share all the vertices,
          // send the entity to the sharing processes.

          // Sort global vertex indices into order
          std::vector<std::int64_t> g_index;
          for (int i = 0; i < num_entity_vertices; ++i)
            g_index.push_back(global_vertex_indices[v[i]]);
          std::sort(g_index.begin(), g_index.end());

          // Record processes this entity belongs to (possibly)
          shared_entities.insert(
              {e, std::set<int>(sharing_procs.begin(), sharing_procs.end())});

          // Record mapping from vertex global indices to local entity index
          entity_to_local_index.insert({g_index, e});

          // Send all possible entities to remote processes
          for (auto p : sharing_procs)
          {
            send_local_entities[p].push_back(e);
            send_entities[p].insert(send_entities[p].end(), g_index.begin(),
                                    g_index.end());
          }
        }
      }
    }

    dolfinx::MPI::all_to_all(comm, send_entities, recv_entities);

    // Check off received entities against sent entities
    // (any which don't match need to be revised).
    // For every shared entity that is sent to another process, the same entity
    // should be returned. If not, it does not exist on the remote process... On
    // the other hand, if an entity is received which has not been sent, then it
    // can be safely ignored.

    // Convert received data back to local index (where possible, otherwise put
    // -1 to ignore)
    for (int p = 0; p < mpi_size; ++p)
    {
      const std::vector<std::int64_t>& recv_p = recv_entities[p];
      for (std::size_t i = 0; i < recv_p.size(); i += num_entity_vertices)
      {
        std::vector<std::int64_t> q(recv_p.begin() + i,
                                    recv_p.begin() + i + num_entity_vertices);

        const auto map_it = entity_to_local_index.find(q);
        if (map_it != entity_to_local_index.end())
          recv_local_entities[p].push_back(map_it->second);
        else
          recv_local_entities[p].push_back(-1);
      }
    }

    // End of scoped region
  }

  // Compare sent and received entities
  for (int p = 0; p < mpi_size; ++p)
  {
    std::vector<std::int32_t> send_p = send_local_entities[p];
    std::vector<std::int32_t> recv_p = recv_local_entities[p];
    std::sort(send_p.begin(), send_p.end());
    std::sort(recv_p.begin(), recv_p.end());

    // If received and sent are identical, there is nothing to do here.
    if (send_p == recv_p)
      continue;

    // Get list of items in send_p, but not in recv_p
    std::vector<std::int32_t> diff;
    std::set_difference(send_p.begin(), send_p.end(), recv_p.begin(),
                        recv_p.end(), std::back_inserter(diff));

    // any entities listed in diff do not exist on process 'p'
    for (auto& q : diff)
    {
      const auto emap_it = shared_entities.find(q);
      assert(emap_it != shared_entities.end());
      int n = emap_it->second.erase(p);
      assert(n == 1);

      // If this was the last sharing process, then this is no longer a shared
      // entity
      if (emap_it->second.empty())
        shared_entities.erase(emap_it);
    }
  }

  // We now know which entities are shared (and with which processes).
  // Three categories: owned, owned+shared, remote+shared=ghost

  global_entity_indices.resize(size, -1);

  // Start numbering

  // Calculate all locally owned entities, and get a local offset
  std::int32_t num_local = size - shared_entities.size();
  for (const auto& q : shared_entities)
  {
    if (*q.second.begin() > mpi_rank)
      ++num_local;
  }

  const std::int64_t local_offset
      = dolfinx::MPI::global_offset(comm, num_local, true);
  std::int64_t n = local_offset;

  // Owned
  for (int i = 0; i < size; ++i)
  {
    const auto it = shared_entities.find(i);
    if (it == shared_entities.end())
    {
      // Owned but not shared
      global_entity_indices[i] = n;
      ++n;
    }
    else if (*(it->second.begin()) > mpi_rank)
    {
      // Owned and shared
      global_entity_indices[it->first] = n;
      ++n;
    }
  }

  assert(n == local_offset + num_local);

  // Now send/recv global indices to/from remotes
  std::vector<std::vector<std::int64_t>> send_indices(mpi_size);
  std::vector<std::vector<std::int64_t>> recv_indices(mpi_size);

  // Revisit the entities we sent out before, sending out our
  // new global indices (only to higher rank processes)
  for (int p = mpi_rank + 1; p < mpi_size; ++p)
  {
    const std::vector<std::int32_t>& send_p = send_local_entities[p];
    for (auto q : send_p)
      send_indices[p].push_back(global_entity_indices[q]);
  }

  dolfinx::MPI::all_to_all(comm, send_indices, recv_indices);

  // Revisit the entities we received before, now with the global
  // index from remote
  for (int p = 0; p < mpi_rank; ++p)
  {
    const std::vector<std::int32_t>& recv_p = recv_local_entities[p];
    for (std::size_t i = 0; i < recv_p.size(); ++i)
    {
      const std::int32_t local_index = recv_p[i];
      // Make sure this is a valid entity, and coming from the owner
      if (local_index != -1 and p == *shared_entities[local_index].begin())
        global_entity_indices[local_index] = recv_indices[p][i];
    }
  }

  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> ghosts(
      global_entity_indices.size() - num_local);
  std::copy(global_entity_indices.begin() + num_local,
            global_entity_indices.end(), ghosts.data());

  std::shared_ptr<common::IndexMap> index_map
      = std::make_shared<common::IndexMap>(comm, num_local, ghosts, 1);

  return std::tuple(std::move(global_entity_indices),
                    std::move(shared_entities), index_map);
}
//-----------------------------------------------------------------------------
template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
reorder_values_by_global_indices(
    MPI_Comm mpi_comm,
    const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& values,
    const std::vector<std::int64_t>& global_indices)
{
  dolfinx::common::Timer t("DistributedMeshTools: reorder values");

  // Number of items to redistribute
  const std::size_t num_local_indices = global_indices.size();
  assert(num_local_indices == (std::size_t)values.rows());

  // Calculate size of overall global vector by finding max index value
  // anywhere
  const std::size_t global_vector_size
      = dolfinx::MPI::max(mpi_comm, *std::max_element(global_indices.begin(),
                                                      global_indices.end()))
        + 1;

  // Send unwanted values off process
  const std::size_t mpi_size = dolfinx::MPI::size(mpi_comm);
  std::vector<std::vector<std::size_t>> indices_to_send(mpi_size);
  std::vector<std::vector<T>> values_to_send(mpi_size);

  // Go through local vector and append value to the appropriate list to
  // send to correct process
  for (std::size_t i = 0; i != num_local_indices; ++i)
  {
    const std::size_t global_i = global_indices[i];
    const std::size_t process_i
        = dolfinx::MPI::index_owner(mpi_comm, global_i, global_vector_size);
    indices_to_send[process_i].push_back(global_i);
    values_to_send[process_i].insert(values_to_send[process_i].end(),
                                     values.row(i).data(),
                                     values.row(i).data() + values.cols());
  }

  // Redistribute the values to the appropriate process - including
  // self. All values are "in the air" at this point. Receive into flat
  // arrays.
  std::vector<std::size_t> received_indices;
  std::vector<T> received_values;
  dolfinx::MPI::all_to_all(mpi_comm, indices_to_send, received_indices);
  dolfinx::MPI::all_to_all(mpi_comm, values_to_send, received_values);

  // Map over received values as Eigen array
  assert(received_indices.size() * values.cols() == received_values.size());
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      received_values_array(received_values.data(), received_indices.size(),
                            values.cols());

  // Create array for new data. Note that any indices which are not
  // received will be uninitialised.
  const std::array<std::int64_t, 2> range
      = dolfinx::MPI::local_range(mpi_comm, global_vector_size);
  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> new_values(
      range[1] - range[0], values.cols());

  // Go through received data in descending order, and place in local
  // partition of the global vector. Any duplicate data (with same
  // index) will be overwritten by values from the lowest rank process.
  for (std::int32_t j = received_indices.size() - 1; j >= 0; --j)
  {
    const std::int64_t global_i = received_indices[j];
    assert(global_i >= range[0] && global_i < range[1]);
    new_values.row(global_i - range[0]) = received_values_array.row(j);
  }

  return new_values;
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
void DistributedMeshTools::number_entities(MPI_Comm comm,
                                           const Topology& topology,
                                           const mesh::CellType cell_type,
                                           int d)
{
  common::Timer timer("Number distributed mesh entities");

  // Return if global entity indices have already been calculated
  if (topology.global_indices(d).size() > 0)
    return;

  // Const-cast to allow data to be attached
  Topology& _topology = const_cast<Topology&>(topology);

  if (dolfinx::MPI::size(comm) == 1)
  {
    if (d != 0 and !topology.connectivity(d, 0))
    {
      throw std::runtime_error(
          "Cannot globally number mesh entities. Local entities of dimension "
          + std::to_string(d) + " have not been computed.");
    }

    // Get number of mesh entities of dimension d on this process
    assert(topology.connectivity(d, 0));
    const std::int32_t size_d = topology.connectivity(d, 0)->num_nodes();

    // Set global entity numbers in mesh
    std::vector<std::int64_t> global_indices(size_d, 0);
    std::iota(global_indices.begin(), global_indices.end(), 0);
    _topology.set_global_indices(d, global_indices);

    // Set IndexMap
    Eigen::Array<std::int64_t, Eigen::Dynamic, 1> ghosts(0);
    auto index_map
        = std::make_shared<common::IndexMap>(comm, size_d, ghosts, 1);
    _topology.set_index_map(d, index_map);

    return;
  }

  // Number entities
  const auto [global_entity_indices, shared_entities, index_map]
      = compute_entity_numbering(comm, topology, cell_type, d);

  // Set IndexMap
  _topology.set_index_map(d, index_map);

  // Set shared entities
  _topology.set_shared_entities(d, shared_entities);

  // Set global entity numbers in mesh
  _topology.set_global_indices(d, global_entity_indices);
}
//-----------------------------------------------------------------------------
void DistributedMeshTools::init_facet_cell_connections(MPI_Comm comm,
                                                       Topology& topology)
{
  // Topological dimension
  const int D = topology.dim();

  if (!topology.connectivity(D - 1, 0))
    throw std::runtime_error("Facet entities have not been computed.");
  if (!topology.connectivity(D - 1, D))
    throw std::runtime_error("Facet-cell connectivity has not been computed.");

  // Calculate the number of global cells attached to each facet
  // essentially defining the exterior surface
  // FIXME: should this be done earlier, e.g. at partitioning stage
  // when dual graph is built?

  // Create vector to hold number of cells connected to each
  // facet. Initially copy over from local values.

  assert(topology.connectivity(D - 1, 0));
  const int num_facets = topology.connectivity(D - 1, 0)->num_nodes();
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> num_global_neighbors(
      num_facets);

  const std::map<std::int32_t, std::set<std::int32_t>>& shared_facets
      = topology.shared_entities(D - 1);

  // Check if no ghost cells
  assert(topology.index_map(D));
  if (topology.index_map(D)->num_ghosts() == 0)
  {
    // Copy local values
    assert(topology.connectivity(D - 1, D));
    auto connectivity = topology.connectivity(D - 1, D);
    for (int f = 0; f < num_facets; ++f)
      num_global_neighbors[f] = connectivity->num_links(f);

    // All shared facets must have two cells, if no ghost cells
    for (const auto& f_it : shared_facets)
      num_global_neighbors[f_it.first] = 2;
  }
  else
  {
    // With ghost cells, shared facets may be on an external edge, so
    // need to check connectivity with the cell owner.

    const std::int32_t mpi_size = MPI::size(comm);
    std::vector<std::vector<std::size_t>> send_facet(mpi_size);
    std::vector<std::vector<std::size_t>> recv_facet(mpi_size);

    // Map shared facets
    std::map<std::size_t, std::size_t> global_to_local_facet;

    const Eigen::Array<int, Eigen::Dynamic, 1>& cell_owners
        = topology.index_map(D)->ghost_owners();
    const std::int32_t ghost_offset_c = topology.index_map(D)->size_local();
    const std::int32_t ghost_offset_f = topology.index_map(D - 1)->size_local();
    const std::map<std::int32_t, std::set<std::int32_t>>& sharing_map_f
        = topology.shared_entities(D - 1);
    const auto& global_facets = topology.global_indices(D - 1);
    assert(topology.connectivity(D - 1, D));
    auto connectivity = topology.connectivity(D - 1, D);
    for (int f = 0; f < num_facets; ++f)
    {
      // Insert shared facets into mapping
      if (sharing_map_f.find(f) != sharing_map_f.end())
        global_to_local_facet.insert({global_facets[f], f});

      // Copy local values
      const int n_cells = connectivity->num_links(f);
      num_global_neighbors[f] = n_cells;

      if ((f >= ghost_offset_f) and n_cells == 1)
      {
        // Singly attached ghost facet - check with owner of attached
        // cell
        auto c = connectivity->links(f);
        assert(c[0] >= ghost_offset_c);
        const int owner = cell_owners[c[0] - ghost_offset_c];
        send_facet[owner].push_back(global_facets[f]);
      }
    }

    MPI::all_to_all(comm, send_facet, recv_facet);

    // Convert received global facet index into number of attached cells
    // and return to sender
    std::vector<std::vector<std::size_t>> send_response(mpi_size);
    for (std::int32_t p = 0; p < mpi_size; ++p)
    {
      for (auto r = recv_facet[p].begin(); r != recv_facet[p].end(); ++r)
      {
        auto map_it = global_to_local_facet.find(*r);
        assert(map_it != global_to_local_facet.end());
        const int n_cells = connectivity->num_links(map_it->second);
        send_response[p].push_back(n_cells);
      }
    }

    MPI::all_to_all(comm, send_response, recv_facet);

    // Insert received result into same facet that it came from
    for (std::int32_t p = 0; p < mpi_size; ++p)
    {
      for (std::size_t i = 0; i < recv_facet[p].size(); ++i)
      {
        auto f_it = global_to_local_facet.find(send_facet[p][i]);
        assert(f_it != global_to_local_facet.end());
        num_global_neighbors[f_it->second] = recv_facet[p][i];
      }
    }
  }

  assert(topology.connectivity(D - 1, D));
  topology.set_global_size({D - 1, D}, num_global_neighbors);
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
DistributedMeshTools::reorder_by_global_indices(
    MPI_Comm mpi_comm,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& values,
    const std::vector<std::int64_t>& global_indices)
{
  return reorder_values_by_global_indices<double>(mpi_comm, values,
                                                  global_indices);
}
//-----------------------------------------------------------------------------
Eigen::Array<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic,
             Eigen::RowMajor>
DistributedMeshTools::reorder_by_global_indices(
    MPI_Comm mpi_comm,
    const Eigen::Ref<const Eigen::Array<std::complex<double>, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>&
        values,
    const std::vector<std::int64_t>& global_indices)
{
  return reorder_values_by_global_indices<std::complex<double>>(
      mpi_comm, values, global_indices);
}
//-----------------------------------------------------------------------------
