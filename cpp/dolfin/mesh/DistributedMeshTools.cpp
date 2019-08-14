// Copyright (C) 2011-2019 Garth N. Wells and Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DistributedMeshTools.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshFunction.h"
#include "MeshIterator.h"
#include "cell_types.h"
#include "dolfin/common/MPI.h"
#include "dolfin/common/Timer.h"
#include "dolfin/graph/Graph.h"
#include "dolfin/graph/SCOTCH.h"
#include <Eigen/Dense>
#include <complex>
#include <dolfin/common/log.h>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
namespace
{
//-----------------------------------------------------------------------------
template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
reorder_values_by_global_indices(
    MPI_Comm mpi_comm,
    const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& values,
    const std::vector<std::int64_t>& global_indices)
{
  dolfin::common::Timer t("DistributedMeshTools: reorder values");

  // Number of items to redistribute
  const std::size_t num_local_indices = global_indices.size();
  assert(num_local_indices == (std::size_t)values.rows());

  // Calculate size of overall global vector by finding max index value
  // anywhere
  const std::size_t global_vector_size
      = dolfin::MPI::max(mpi_comm, *std::max_element(global_indices.begin(),
                                                     global_indices.end()))
        + 1;

  // Send unwanted values off process
  const std::size_t mpi_size = dolfin::MPI::size(mpi_comm);
  std::vector<std::vector<std::size_t>> indices_to_send(mpi_size);
  std::vector<std::vector<T>> values_to_send(mpi_size);

  // Go through local vector and append value to the appropriate list to
  // send to correct process
  for (std::size_t i = 0; i != num_local_indices; ++i)
  {
    const std::size_t global_i = global_indices[i];
    const std::size_t process_i
        = dolfin::MPI::index_owner(mpi_comm, global_i, global_vector_size);
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
  dolfin::MPI::all_to_all(mpi_comm, indices_to_send, received_indices);
  dolfin::MPI::all_to_all(mpi_comm, values_to_send, received_values);

  // Map over received values as Eigen array
  assert(received_indices.size() * values.cols() == received_values.size());
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      received_values_array(received_values.data(), received_indices.size(),
                            values.cols());

  // Create array for new data. Note that any indices which are not
  // received will be uninitialised.
  const std::array<std::int64_t, 2> range
      = dolfin::MPI::local_range(mpi_comm, global_vector_size);
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
void sort_array_by_row(Eigen::Ref<Eigen::Array<std::int64_t, Eigen::Dynamic,
                                               Eigen::Dynamic, Eigen::RowMajor>>
                           array)
{
  // Find the permutation that sorts the rows into order
  std::vector<int> perm(array.rows());
  std::iota(perm.begin(), perm.end(), 0);
  std::sort(perm.begin(), perm.end(), [&array](int i, int j) {
    return std::lexicographical_compare(
        array.row(i).data(), array.row(i).data() + array.cols(),
        array.row(j).data(), array.row(j).data() + array.cols());
  });

  // Apply the permutation to the array
  Eigen::Map<Eigen::PermutationMatrix<Eigen::Dynamic>> perm_map(perm.data(),
                                                                array.rows());
  array = perm_map.inverse() * array.matrix();
}

} // namespace

//-----------------------------------------------------------------------------
void DistributedMeshTools::number_entities(const Mesh& mesh, int d)
{
  common::Timer timer("Number distributed mesh entities");

  // Return if global entity indices have already been calculated
  if (mesh.topology().have_global_indices(d))
    return;

  // Const-cast to allow data to be attached
  Mesh& _mesh = const_cast<Mesh&>(mesh);

  if (dolfin::MPI::size(mesh.mpi_comm()) == 1)
  {
    // Set global entity numbers in mesh
    mesh.create_entities(d);
    _mesh.topology().set_num_entities_global(d, mesh.num_entities(d));
    std::vector<std::int64_t> global_indices(mesh.num_entities(d), 0);
    std::iota(global_indices.begin(), global_indices.end(), 0);
    _mesh.topology().set_global_indices(d, global_indices);
    return;
  }

  // Get shared entities map
  std::map<std::int32_t, std::set<std::int32_t>>& shared_entities
      = _mesh.topology().shared_entities(d);

  // Number entities
  std::vector<std::int64_t> global_entity_indices;
  std::size_t num_global_entities;
  std::tie(global_entity_indices, shared_entities, num_global_entities)
      = number_entities_computation(mesh, d);

  // Set global entity numbers in mesh
  _mesh.topology().set_num_entities_global(d, num_global_entities);
  _mesh.topology().set_global_indices(d, global_entity_indices);
}
//-----------------------------------------------------------------------------
std::tuple<std::vector<std::int64_t>,
           std::map<std::int32_t, std::set<std::int32_t>>, std::size_t>
DistributedMeshTools::number_entities_computation(const Mesh& mesh, int d)
{
  LOG(INFO)
      << "Number mesh entities for distributed mesh (for specified vertex ids)."
      << d;
  common::Timer timer(
      "Number mesh entities for distributed mesh (for specified vertex ids)");

  std::vector<std::int64_t> global_entity_indices;
  std::map<std::int32_t, std::set<std::int32_t>> shared_entities;

  // Check that we're not re-numbering vertices (these are fixed at mesh
  // construction)
  if (d == 0)
  {
    throw std::runtime_error(
        "Global vertex indices exist at input. Cannot be renumbered");
  }

  if (d == mesh.topology().dim())
  {
    // Numbering cells.
    // FIXME: Should be redundant?
    shared_entities.clear();
    global_entity_indices = mesh.topology().global_indices(d);
    return std::make_tuple(std::move(global_entity_indices),
                           std::move(shared_entities),
                           mesh.num_entities_global(d));
  }

  // MPI communicator
  const MPI_Comm mpi_comm = mesh.mpi_comm();

  // Get number of processes and process number
  const int mpi_size = MPI::size(mpi_comm);
  const int mpi_rank = MPI::rank(mpi_comm);

  // Initialize entities of dimension d locally
  mesh.create_entities(d);

  // Get vertex global indices
  const std::vector<std::int64_t>& global_vertex_indices
      = mesh.topology().global_indices(0);

  // Get shared vertices (local index, [sharing processes])
  // already determined in Mesh distribution
  const std::map<std::int32_t, std::set<std::int32_t>>& shared_vertices_local
      = mesh.topology().shared_entities(0);

  // Make a mask for shared vertices (true if shared)
  std::vector<bool> vertex_shared(mesh.num_entities(0), false);
  for (const auto& v : shared_vertices_local)
    vertex_shared[v.first] = true;

  // Send and receive shared entities
  std::vector<std::vector<std::int64_t>> send_entities(mpi_size);
  std::vector<std::vector<std::int64_t>> recv_entities(mpi_size);
  std::map<std::vector<std::int64_t>, std::int32_t> entity_to_local_index;

  // Get number of vertices in this entity
  const mesh::CellType& ct = mesh.cell_type;
  const mesh::CellType& et = mesh::cell_entity_type(ct, d);
  const int num_entity_vertices = mesh::num_cell_vertices(et);

  for (auto& e : mesh::MeshRange(mesh, d, mesh::MeshRangeType::ALL))
  {
    const std::size_t local_index = e.index();
    const std::int32_t* v = e.entities(0);

    // Entity can only be shared if all vertices are shared
    // but this is not a sufficient condition
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

      std::vector<std::int64_t> g_index = {global_vertex_indices[v[0]]};

      for (int i = 1; i < num_entity_vertices; ++i)
      {
        g_index.push_back(global_vertex_indices[v[i]]);

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
        std::sort(g_index.begin(), g_index.end());

        // Record processes this entity belongs to (possibly)
        shared_entities.insert(
            {local_index,
             std::set<int>(sharing_procs.begin(), sharing_procs.end())});

        // Record mapping from vertex global indices to local entity index
        entity_to_local_index.insert({g_index, local_index});

        // Send all possible entities to remote processes
        for (auto p : sharing_procs)
        {
          send_entities[p].insert(send_entities[p].end(), g_index.begin(),
                                  g_index.end());
        }
      }
    }
  }

  // Sort sending data into ascending order by row.
  // Useful later for set_difference, and better to do before sending.
  for (int p = 0; p < mpi_size; ++p)
  {
    int num_send = send_entities[p].size() / num_entity_vertices;
    // Map over data with Eigen Array
    Eigen::Map<Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::RowMajor>>
        send_data(send_entities[p].data(), num_send, num_entity_vertices);

    sort_array_by_row(send_data);
  }

  MPI::all_to_all(mpi_comm, send_entities, recv_entities);

  // Check off received entities against sent entities
  // (any which don't match need to be revised).
  // For every shared entity that is sent to another process, the same entity
  // should be returned. If not, it does not exist on the remote process... If
  // an entity is received which has not been sent, then it can be ignored.

  for (int p = 0; p < mpi_size; ++p)
  {
    const std::vector<std::int64_t>& sent_p = send_entities[p];
    const std::vector<std::int64_t>& recv_p = recv_entities[p];

    // If received and sent are identical, there is nothing to do here.
    if (sent_p == recv_p)
      continue;

    std::vector<std::vector<std::int64_t>> sent_as_ent;
    std::vector<std::vector<std::int64_t>> recv_as_ent;
    for (std::size_t i = 0; i < sent_p.size(); i += num_entity_vertices)
      sent_as_ent.push_back(std::vector<std::int64_t>(
          sent_p.begin() + i, sent_p.begin() + i + num_entity_vertices));
    for (std::size_t i = 0; i < recv_p.size(); i += num_entity_vertices)
      recv_as_ent.push_back(std::vector<std::int64_t>(
          recv_p.begin() + i, recv_p.begin() + i + num_entity_vertices));

    std::vector<std::vector<std::int64_t>> diff;

    // Get list of items in sent, but not in recv
    std::set_difference(sent_as_ent.begin(), sent_as_ent.end(),
                        recv_as_ent.begin(), recv_as_ent.end(),
                        std::back_inserter(diff));

    // any entities listed in diff do not exist on process 'p'
    for (auto& q : diff)
    {
      const auto map_it = entity_to_local_index.find(q);
      assert(map_it != entity_to_local_index.end());
      const std::int32_t entity_idx = map_it->second;
      const auto emap_it = shared_entities.find(entity_idx);
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

  // Start numbering
  global_entity_indices.resize(mesh.num_entities(d), -1);

  std::size_t num_local = mesh.num_entities(d) - shared_entities.size();
  for (const auto& q : shared_entities)
  {
    if (*q.second.begin() > mpi_rank)
      ++num_local;
  }

  const std::int64_t local_offset
      = MPI::global_offset(mpi_comm, num_local, true);
  const std::int64_t num_global = MPI::sum(mpi_comm, num_local);
  std::int64_t n = local_offset;

  // Owned
  for (int i = 0; i < mesh.num_entities(d); ++i)
  {
    const auto it = shared_entities.find(i);
    if (it == shared_entities.end())
    {
      global_entity_indices[i] = n;
      ++n;
    }
  }

  // Owned and shared
  for (const auto& q : shared_entities)
  {
    if (*q.second.begin() > mpi_rank)
    {
      global_entity_indices[q.first] = n;
      ++n;
    }
  }

  // Now send/recv global indices to/from remotes
  std::vector<std::vector<std::int64_t>> send_indices(mpi_size);
  std::vector<std::vector<std::int64_t>> recv_indices(mpi_size);

  // Revisit the entities we sent out before, sending out our
  // new global indices (only to higher rank processes)
  for (int p = mpi_rank + 1; p < mpi_size; ++p)
  {
    const std::vector<std::int64_t>& sent_p = send_entities[p];
    for (std::size_t i = 0; i < sent_p.size(); i += num_entity_vertices)
    {
      std::vector<std::int64_t> q(sent_p.begin() + i,
                                  sent_p.begin() + i + num_entity_vertices);
      const auto map_it = entity_to_local_index.find(q);
      assert(map_it != entity_to_local_index.end());
      int local_index = map_it->second;
      send_indices[p].push_back(global_entity_indices[local_index]);
    }
  }

  MPI::all_to_all(mpi_comm, send_indices, recv_indices);

  // Revisit the entities we received before, now with the global
  // index from remote
  for (int p = 0; p < mpi_size; ++p)
  {
    const std::vector<std::int64_t>& recv_p = recv_entities[p];
    for (std::size_t i = 0; i < recv_indices[p].size(); ++i)
    {
      std::vector<std::int64_t> q(recv_p.begin() + i * num_entity_vertices,
                                  recv_p.begin()
                                      + (i + 1) * num_entity_vertices);

      const auto map_it = entity_to_local_index.find(q);
      // Make sure this is a valid entity, and coming from the owner
      if (map_it != entity_to_local_index.end()
          and p == *shared_entities[map_it->second].begin())
      {
        int local_index = map_it->second;
        global_entity_indices[local_index] = recv_indices[p][i];
      }
    }
  }

  return std::make_tuple(std::move(global_entity_indices),
                         std::move(shared_entities), num_global);
}
//-----------------------------------------------------------------------------
std::map<std::size_t, std::set<std::pair<std::size_t, std::size_t>>>
DistributedMeshTools::locate_off_process_entities(
    const std::vector<std::size_t>& entity_indices, std::size_t dim,
    const Mesh& mesh)
{
  common::Timer timer("Locate off-process entities");

  if (dim == 0)
  {
    LOG(WARNING)
        << "DistributedMeshTools::host_processes has not been tested for "
           "vertices.";
  }

  // Mesh topology dim
  const std::size_t D = mesh.topology().dim();

  // Check that entity is a vertex or a cell
  if (dim != 0 && dim != D)
  {
    throw std::runtime_error(
        "This version of DistributedMeshTools::host_processes is only for "
        "vertices or cells");
  }

  // Check that global numbers have been computed.
  if (!mesh.topology().have_global_indices(dim)
      or !mesh.topology().have_global_indices(D))
  {
    throw std::runtime_error(
        "Global mesh entity numbers have not been computed");
  }

  // Get global cell entity indices on this process
  const std::vector<std::int64_t> global_entity_indices
      = mesh.topology().global_indices(dim);

  assert((std::int64_t)global_entity_indices.size() == mesh.num_entities(D));

  // Prepare map to hold process numbers
  std::map<std::size_t, std::set<std::pair<std::size_t, std::size_t>>>
      processes;

  // FIXME: work on optimising below code

  // List of indices to send
  std::vector<std::size_t> my_entities;

  // Remove local cells from my_entities to reduce communication
  if (dim == D)
  {
    // In order to fill vector my_entities...
    // build and populate a local set for non-local cells
    std::set<std::size_t> set_of_my_entities(entity_indices.begin(),
                                             entity_indices.end());

    const std::map<std::int32_t, std::set<std::int32_t>>& sharing_map
        = mesh.topology().shared_entities(D);

    // FIXME: This can be made more efficient by exploiting fact that
    //        set is sorted
    // Remove local cells from set_of_my_entities to reduce communication
    for (std::size_t j = 0; j < global_entity_indices.size(); ++j)
    {
      if (sharing_map.find(j) == sharing_map.end())
        set_of_my_entities.erase(global_entity_indices[j]);
    }
    // Copy entries from set_of_my_entities to my_entities
    my_entities = std::vector<std::size_t>(set_of_my_entities.begin(),
                                           set_of_my_entities.end());
  }
  else
    my_entities = entity_indices;

  // FIXME: handle case when my_entities.empty()
  // assert(!my_entities.empty());

  // Prepare data structures for send/receive
  const MPI_Comm mpi_comm = mesh.mpi_comm();
  const std::size_t num_proc = MPI::size(mpi_comm);
  const std::size_t proc_num = MPI::rank(mpi_comm);
  const std::size_t max_recv = MPI::max(mpi_comm, my_entities.size());
  std::vector<std::size_t> off_process_entities(max_recv);

  // Send and receive data
  for (std::size_t k = 1; k < num_proc; ++k)
  {
    const std::size_t src = (proc_num - k + num_proc) % num_proc;
    const std::size_t dest = (proc_num + k) % num_proc;
    MPI::send_recv(mpi_comm, my_entities, dest, off_process_entities, src);

    const std::size_t recv_entity_count = off_process_entities.size();

    // Check if this process owns received entities, and if so
    // store local index
    std::vector<std::size_t> my_hosted_entities;
    {
      // Build a temporary map hosting global_entity_indices
      std::map<std::size_t, std::size_t> map_of_global_entity_indices;
      for (std::size_t j = 0; j < global_entity_indices.size(); j++)
        map_of_global_entity_indices[global_entity_indices[j]] = j;

      for (std::size_t j = 0; j < recv_entity_count; j++)
      {
        // Check if this process hosts 'received_entity'
        const std::size_t received_entity = off_process_entities[j];
        std::map<std::size_t, std::size_t>::const_iterator it
            = map_of_global_entity_indices.find(received_entity);
        if (it != map_of_global_entity_indices.end())
        {
          const std::size_t local_index = it->second;
          my_hosted_entities.push_back(received_entity);
          my_hosted_entities.push_back(local_index);
        }
      }
    }

    // Send/receive hosted cells
    const std::size_t max_recv_host_proc
        = MPI::max(mpi_comm, my_hosted_entities.size());
    std::vector<std::size_t> host_processes(max_recv_host_proc);
    MPI::send_recv(mpi_comm, my_hosted_entities, src, host_processes, dest);

    const std::size_t recv_hostproc_count = host_processes.size();
    for (std::size_t j = 0; j < recv_hostproc_count; j += 2)
    {
      const std::size_t global_index = host_processes[j];
      const std::size_t local_index = host_processes[j + 1];
      processes[global_index].insert({dest, local_index});
    }

    // FIXME: Do later for efficiency
    // Remove entries from entities (from my_entities) that cannot
    // reside on more processes (i.e., cells)
  }

  // Sanity check
  const std::set<std::size_t> test_set(my_entities.begin(), my_entities.end());
  const std::size_t number_expected = test_set.size();
  if (number_expected != processes.size())
    throw std::runtime_error("Sanity check failed");

  return processes;
}
//-----------------------------------------------------------------------------
std::unordered_map<std::int32_t,
                   std::vector<std::pair<std::int32_t, std::int32_t>>>
DistributedMeshTools::compute_shared_entities(const Mesh& mesh, std::size_t d)
{
  LOG(INFO) << "Compute shared mesh entities of dimension" << d;
  common::Timer timer("Computed shared mesh entities");

  // MPI communicator
  const MPI_Comm mpi_comm = mesh.mpi_comm();
  const int comm_size = MPI::size(mpi_comm);

  // Return empty set if running in serial
  if (MPI::size(mpi_comm) == 1)
  {
    return std::unordered_map<
        std::int32_t, std::vector<std::pair<std::int32_t, std::int32_t>>>();
  }

  // Initialize entities of dimension d
  mesh.create_entities(d);

  // Number entities (globally)
  number_entities(mesh, d);

  // Get shared entities to processes map
  const std::map<std::int32_t, std::set<std::int32_t>>& shared_entities
      = mesh.topology().shared_entities(d);

  // Get local-to-global indices map
  const std::vector<std::int64_t>& global_indices_map
      = mesh.topology().global_indices(d);

  // Global-to-local map for each process
  std::unordered_map<std::size_t, std::unordered_map<std::size_t, std::size_t>>
      global_to_local;

  // Pack global indices for sending to sharing processes
  std::vector<std::vector<std::size_t>> send_indices(comm_size);
  std::vector<std::vector<std::size_t>> local_sent_indices(comm_size);
  for (auto shared_entity = shared_entities.cbegin();
       shared_entity != shared_entities.cend(); ++shared_entity)
  {
    // Local index
    const std::int32_t local_index = shared_entity->first;

    // Global index
    assert(local_index < (std::int32_t)global_indices_map.size());
    std::size_t global_index = global_indices_map[local_index];

    // Destination process
    const std::set<std::int32_t>& sharing_processes = shared_entity->second;

    // Pack data for sending and build global-to-local map
    for (auto dest = sharing_processes.cbegin();
         dest != sharing_processes.cend(); ++dest)
    {
      send_indices[*dest].push_back(global_index);
      local_sent_indices[*dest].push_back(local_index);
      global_to_local[*dest].insert({global_index, local_index});
    }
  }

  std::vector<std::vector<std::size_t>> recv_entities;
  MPI::all_to_all(mpi_comm, send_indices, recv_entities);

  // Clear send data
  send_indices.clear();
  send_indices.resize(comm_size);

  // Determine local entities indices for received global entity indices
  for (std::size_t p = 0; p < recv_entities.size(); ++p)
  {
    // Get process number of neighbour
    const std::size_t sending_proc = p;

    if (recv_entities[p].size() > 0)
    {
      // Get global-to-local map for neighbour process
      std::unordered_map<
          std::size_t,
          std::unordered_map<std::size_t, std::size_t>>::const_iterator it
          = global_to_local.find(sending_proc);
      assert(it != global_to_local.end());
      const std::unordered_map<std::size_t, std::size_t>&
          neighbour_global_to_local
          = it->second;

      // Build vector of local indices
      const std::vector<std::size_t>& global_indices_recv = recv_entities[p];
      for (std::size_t i = 0; i < global_indices_recv.size(); ++i)
      {
        // Global index
        const std::size_t global_index = global_indices_recv[i];

        // Find local index corresponding to global index
        std::unordered_map<std::size_t, std::size_t>::const_iterator
            n_global_to_local
            = neighbour_global_to_local.find(global_index);

        assert(n_global_to_local != neighbour_global_to_local.end());
        const std::size_t my_local_index = n_global_to_local->second;
        send_indices[sending_proc].push_back(my_local_index);
      }
    }
  }

  MPI::all_to_all(mpi_comm, send_indices, recv_entities);

  // Build map
  std::unordered_map<std::int32_t,
                     std::vector<std::pair<std::int32_t, std::int32_t>>>
      shared_local_indices_map;

  // Loop over data received from each process
  for (std::size_t p = 0; p < recv_entities.size(); ++p)
  {
    if (recv_entities[p].size() > 0)
    {
      // Process that shares entities
      const std::size_t proc = p;

      // Local indices on sharing process
      const std::vector<std::size_t>& neighbour_local_indices
          = recv_entities[p];

      // Local indices on this process
      const std::vector<std::size_t>& my_local_indices = local_sent_indices[p];

      // Check that sizes match
      assert(my_local_indices.size() == neighbour_local_indices.size());

      for (std::size_t i = 0; i < neighbour_local_indices.size(); ++i)
      {
        shared_local_indices_map[my_local_indices[i]].push_back(
            {proc, neighbour_local_indices[i]});
      }
    }
  }

  return shared_local_indices_map;
}
//-----------------------------------------------------------------------------
void DistributedMeshTools::init_facet_cell_connections(Mesh& mesh)
{
  // Topological dimension
  const int D = mesh.topology().dim();

  // Initialize entities of dimension d
  mesh.create_entities(D - 1);

  // Initialise local facet-cell connections.
  mesh.create_connectivity(D - 1, D);

  // Global numbering
  number_entities(mesh, D - 1);

  // Calculate the number of global cells attached to each facet
  // essentially defining the exterior surface
  // FIXME: should this be done earlier, e.g. at partitioning stage
  // when dual graph is built?

  // Create vector to hold number of cells connected to each
  // facet. Initially copy over from local values.

  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> num_global_neighbors(
      mesh.num_entities(D - 1));

  std::map<std::int32_t, std::set<std::int32_t>>& shared_facets
      = mesh.topology().shared_entities(D - 1);

  // Check if no ghost cells
  if (mesh.topology().ghost_offset(D) == mesh.topology().size(D))
  {
    // Copy local values
    assert(mesh.topology().connectivity(D - 1, D));
    auto connectivity = mesh.topology().connectivity(D - 1, D);
    for (auto& f : mesh::MeshRange(mesh, D - 1))
      num_global_neighbors[f.index()] = connectivity->size(f.index());

    // All shared facets must have two cells, if no ghost cells
    for (auto f_it = shared_facets.begin(); f_it != shared_facets.end(); ++f_it)
      num_global_neighbors[f_it->first] = 2;
  }
  else
  {
    // With ghost cells, shared facets may be on an external edge, so
    // need to check connectivity with the cell owner.

    const std::int32_t mpi_size = MPI::size(mesh.mpi_comm());
    std::vector<std::vector<std::size_t>> send_facet(mpi_size);
    std::vector<std::vector<std::size_t>> recv_facet(mpi_size);

    // Map shared facets
    std::map<std::size_t, std::size_t> global_to_local_facet;

    const std::vector<std::int32_t>& cell_owners = mesh.topology().owner(D);
    const std::int32_t ghost_offset_c = mesh.topology().ghost_offset(D);
    const std::int32_t ghost_offset_f = mesh.topology().ghost_offset(D - 1);
    const std::map<std::int32_t, std::set<std::int32_t>>& sharing_map_f
        = mesh.topology().shared_entities(D - 1);
    const auto& global_facets = mesh.topology().global_indices(D - 1);
    assert(mesh.topology().connectivity(D - 1, D));
    auto connectivity = mesh.topology().connectivity(D - 1, D);
    for (auto& f : mesh::MeshRange(mesh, D - 1, mesh::MeshRangeType::ALL))
    {
      // Insert shared facets into mapping
      if (sharing_map_f.find(f.index()) != sharing_map_f.end())
        global_to_local_facet.insert({global_facets[f.index()], f.index()});

      // Copy local values
      const int n_cells = connectivity->size(f.index());
      num_global_neighbors[f.index()] = n_cells;

      if ((f.index() >= ghost_offset_f) and n_cells == 1)
      {
        // Singly attached ghost facet - check with owner of attached
        // cell
        assert(f.entities(D)[0] >= ghost_offset_c);
        const int owner = cell_owners[f.entities(D)[0] - ghost_offset_c];
        send_facet[owner].push_back(global_facets[f.index()]);
      }
    }

    MPI::all_to_all(mesh.mpi_comm(), send_facet, recv_facet);

    // Convert received global facet index into number of attached cells
    // and return to sender
    std::vector<std::vector<std::size_t>> send_response(mpi_size);
    for (std::int32_t p = 0; p < mpi_size; ++p)
    {
      for (auto r = recv_facet[p].begin(); r != recv_facet[p].end(); ++r)
      {
        auto map_it = global_to_local_facet.find(*r);
        assert(map_it != global_to_local_facet.end());
        const mesh::MeshEntity local_facet(mesh, D - 1, map_it->second);
        const int n_cells = connectivity->size(map_it->second);
        send_response[p].push_back(n_cells);
      }
    }

    MPI::all_to_all(mesh.mpi_comm(), send_response, recv_facet);

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

  assert(mesh.topology().connectivity(D - 1, D));
  mesh.topology().connectivity(D - 1, D)->set_global_size(num_global_neighbors);
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
