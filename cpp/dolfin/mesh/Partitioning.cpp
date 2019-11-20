// Copyright (C) 2008-2014 Niclas Jansson, Ola Skavhaug, Anders Logg,
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Partitioning.h"
#include "DistributedMeshTools.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshFunction.h"
#include "MeshValueCollection.h"
#include "PartitionData.h"
#include "Topology.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Set.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/log.h>
#include <dolfin/graph/CSRGraph.h>
#include <dolfin/graph/GraphBuilder.h>
#include <dolfin/graph/KaHIP.h>
#include <dolfin/graph/ParMETIS.h>
#include <dolfin/graph/SCOTCH.h>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <set>

using namespace dolfin;
using namespace dolfin::mesh;

namespace
{
//-----------------------------------------------------------------------------
// std::map<std::int64_t, std::set<int>>
// distribute_points_sharing(MPI_Comm mpi_comm,
//                           const std::array<std::int64_t, 2> local_range,
//                           const std::vector<std::int64_t>& global_index,
//                           const std::vector<int>& offsets)
// {
//   common::Timer timer("Distribute shared point information");
//   const int mpi_size = dolfin::MPI::size(mpi_comm);

//   // Map for sharing information
//   std::vector<std::set<int>> point_to_procs_set(local_range[1]
//                                                 - local_range[0]);
//   for (int i = 0; i < mpi_size; ++i)
//   {
//     for (int j = offsets[i]; j < offsets[i + 1]; ++j)
//       point_to_procs_set[global_index[j] - local_range[0]].insert(i);
//   }

//   std::vector<std::vector<std::int64_t>> send_sharing(mpi_size);
//   std::vector<std::int64_t> recv_sharing(mpi_size);
//   for (int i = local_range[0]; i < local_range[1]; ++i)
//   {
//     if (point_to_procs_set[i].size() > 1)
//     {
//       for (int r : point_to_procs_set[i])
//       {
//         send_sharing[r].push_back(point_to_procs_set[i].size() - 1);
//         send_sharing[r].push_back(i);
//         for (int proc : point_to_procs_set[i])
//         {
//           if (proc != r)
//             send_sharing[r].push_back(proc);
//         }
//       }
//     }
//   }
//   dolfin::MPI::all_to_all(mpi_comm, send_sharing, recv_sharing);

//   std::map<std::int64_t, std::set<int>> point_to_procs;
//   for (auto q = recv_sharing.begin(); q < recv_sharing.end(); q += (*q + 2))
//   {
//     const std::int64_t global_index = *(q + 1);
//     std::set<int> procs(q + 2, q + 2 + *q);
//     point_to_procs.insert({global_index, procs});
//   }

//   return point_to_procs;
// }
//-----------------------------------------------------------------------------
// This function takes the partition computed by the partitioner
// (which tells us to which process each of the local cells stored on
// this process belongs) and sends the cells
// to the appropriate owning process. Ghost cells are also sent to all processes
// that need them, along with the list of sharing processes.
//
// Returns (new_cell_vertices, new_global_cell_indices,
// new_cell_partition, shared_cells, number of non-ghost cells on this
// process).
std::tuple<
    Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
    std::vector<std::int64_t>, std::vector<int>,
    std::map<std::int32_t, std::set<std::int32_t>>, std::int32_t>
distribute_cells(
    const MPI_Comm mpi_comm,
    const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>&
        cell_vertices,
    const std::vector<std::int64_t>& global_cell_indices,
    const PartitionData& cell_partition)
{
  // This function takes the partition computed by the partitioner
  // stored in PartitionData cell_partition. Some cells go to multiple
  // destinations. Each cell is transmitted to its final destination(s)
  // including its global index, and the cell owner (for ghost cells this will
  // be different from the destination)

  LOG(INFO) << "Distribute cells during distributed mesh construction";

  common::Timer timer("Distribute cells");

  const std::size_t mpi_size = dolfin::MPI::size(mpi_comm);
  const std::size_t mpi_rank = dolfin::MPI::rank(mpi_comm);

  // Global offset, used to build global cell index, if not given
  std::int64_t global_offset
      = dolfin::MPI::global_offset(mpi_comm, cell_vertices.rows(), true);
  bool build_global_index = global_cell_indices.empty();

  // Get dimensions
  const std::int32_t num_local_cells = cell_vertices.rows();
  const std::int32_t num_cell_vertices = cell_vertices.cols();
  assert(cell_partition.size() == num_local_cells);

  // Send all cells to their destinations including their global
  // indices.  First element of vector is cell count of un-ghosted
  // cells, second element is count of ghost cells.
  std::vector<std::vector<std::size_t>> send_cell_vertices(
      mpi_size, std::vector<std::size_t>(2, 0));

  for (std::int32_t i = 0; i < cell_partition.size(); ++i)
  {
    std::int32_t num_procs = cell_partition.num_procs(i);
    const std::int32_t* sharing_procs = cell_partition.procs(i);
    for (std::int32_t j = 0; j < num_procs; ++j)
    {
      // Create reference to destination vector
      std::vector<std::size_t>& send_cell_dest
          = send_cell_vertices[sharing_procs[j]];

      // Count of ghost cells, followed by ghost processes
      if (num_procs > 1)
      {
        send_cell_dest.push_back(num_procs);
        send_cell_dest.insert(send_cell_dest.end(), sharing_procs,
                              sharing_procs + num_procs);
      }
      else // unghosted
        send_cell_dest.push_back(0);

      // Global cell index
      if (build_global_index)
        send_cell_dest.push_back(global_offset + i);
      else
        send_cell_dest.push_back(global_cell_indices[i]);

      // Global vertex indices
      send_cell_dest.insert(send_cell_dest.end(), cell_vertices.row(i).data(),
                            cell_vertices.row(i).data() + num_cell_vertices);

      // First entry is the owner, so this counts as a 'local' cell
      // subsequent entries are 'remote ghosts'
      if (j == 0)
        send_cell_dest[0]++;
      else
        send_cell_dest[1]++;
    }
  }

  // Distribute cell-vertex connectivity and ownership information
  std::vector<std::vector<std::size_t>> received_cell_vertices(mpi_size);
  dolfin::MPI::all_to_all(mpi_comm, send_cell_vertices, received_cell_vertices);

  // Count number of received cells (first entry in vector) and find out
  // how many ghost cells there are...
  std::size_t local_count = 0;
  std::size_t ghost_count = 0;
  for (std::size_t p = 0; p < mpi_size; ++p)
  {
    std::vector<std::size_t>& received_data = received_cell_vertices[p];
    local_count += received_data[0];
    ghost_count += received_data[1];
  }

  const std::size_t all_count = ghost_count + local_count;

  Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      new_cell_vertices(all_count, num_cell_vertices);
  std::vector<std::int64_t> new_global_cell_indices(all_count);
  std::vector<int> new_cell_partition(all_count);

  // Unpack received data
  // Create a map from cells which are shared, to the remote processes
  // which share them - corral ghost cells to end of range
  std::size_t c = 0;
  std::size_t gc = local_count;
  std::map<std::int32_t, std::set<std::int32_t>> shared_cells;
  for (std::size_t p = 0; p < mpi_size; ++p)
  {
    std::vector<std::size_t>& received_data = received_cell_vertices[p];
    for (auto it = received_data.begin() + 2; it != received_data.end();
         it += (*it + num_cell_vertices + 2))
    {
      auto tmp_it = it;
      const std::int32_t num_ghosts = *tmp_it++;

      // Determine owner, and indexing.
      // Note that *tmp_it may be equal to mpi_rank
      const std::size_t owner = (num_ghosts == 0) ? mpi_rank : *tmp_it;
      const std::size_t idx = (owner == mpi_rank) ? c : gc;

      assert(idx < new_cell_partition.size());
      new_cell_partition[idx] = owner;
      if (num_ghosts != 0)
      {
        std::set<std::int32_t> proc_set(tmp_it, tmp_it + num_ghosts);

        // Remove self from set of sharing processes
        proc_set.erase(mpi_rank);
        shared_cells.insert({idx, proc_set});
        tmp_it += num_ghosts;
      }

      new_global_cell_indices[idx] = *tmp_it++;
      for (std::int32_t j = 0; j < num_cell_vertices; ++j)
        new_cell_vertices(idx, j) = *tmp_it++;

      if (owner == mpi_rank)
        ++c;
      else
        ++gc;
    }
  }

  assert(c == local_count);
  assert(gc == all_count);

  return std::tuple(
      std::move(new_cell_vertices), std::move(new_global_cell_indices),
      std::move(new_cell_partition), std::move(shared_cells), local_count);
}
//-----------------------------------------------------------------------------
// Distribute additional cells implied by connectivity via vertex. The
// input cell_vertices, shared_cells, global_cell_indices and
// cell_partition must already be distributed with a ghost layer by
// shared_facet.
// FIXME: shared_cells, cell_vertices, global_cell_indices and
// cell_partition are all modified by this function.
void distribute_cell_layer(
    MPI_Comm mpi_comm, const int num_regular_cells,
    std::map<std::int32_t, std::set<std::int32_t>>& shared_cells,
    Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        cell_vertices,
    std::vector<std::int64_t>& global_cell_indices,
    std::vector<int>& cell_partition)
{
  common::Timer timer("Distribute cell layer");

  const int mpi_size = dolfin::MPI::size(mpi_comm);
  const int mpi_rank = dolfin::MPI::rank(mpi_comm);

  // Map from shared vertex to the set of cells containing it
  std::map<std::int64_t, std::vector<std::int64_t>> sh_vert_to_cell;

  // Global to local mapping of cell indices
  std::map<std::int64_t, int> cell_global_to_local;

  // Iterate only over ghost cells
  for (Eigen::Index i = num_regular_cells; i < cell_vertices.rows(); ++i)
  {
    // Add map entry for each vertex of ghost cells
    for (Eigen::Index p = 0; p < cell_vertices.cols(); ++p)
    {
      sh_vert_to_cell.insert(
          {cell_vertices(i, p), std::vector<std::int64_t>()});
    }

    cell_global_to_local.insert({global_cell_indices[i], i});
  }

  // Iterate only over regular (non-ghost) cells
  for (int i = 0; i < num_regular_cells; ++i)
  {
    for (Eigen::Index j = 0; j != cell_vertices.cols(); ++j)
    {
      // If vertex already in map, append local cell index to set
      auto vc_it = sh_vert_to_cell.find(cell_vertices(i, j));
      if (vc_it != sh_vert_to_cell.end())
      {
        cell_global_to_local.insert({global_cell_indices[i], i});
        vc_it->second.push_back(i);
      }
    }
  }

  // sh_vert_to_cell now contains a mapping from the vertices of ghost
  // cells to any regular cells which they are also incident with.

  // Send lists of cells/owners to "index owner" of vertex, collating
  // and sending back out...
  std::vector<std::vector<std::int64_t>> send_vertcells(mpi_size);
  std::vector<std::vector<std::int64_t>> recv_vertcells(mpi_size);
  for (const auto& vc_it : sh_vert_to_cell)
  {
    // Generate unique destination "index owner" based on vertex index
    const int dest = (vc_it.first) % mpi_size;

    std::vector<std::int64_t>& sendv = send_vertcells[dest];

    // Pack as [cell_global_index, this_vertex, [other_vertices]]
    for (const auto& q : vc_it.second)
    {
      sendv.push_back(global_cell_indices[q]);
      sendv.push_back(vc_it.first);
      for (Eigen::Index v = 0; v < cell_vertices.cols(); ++v)
      {
        if (cell_vertices(q, v) != vc_it.first)
          sendv.push_back(cell_vertices(q, v));
      }
    }
  }

  dolfin::MPI::all_to_all(mpi_comm, send_vertcells, recv_vertcells);

  const std::int32_t num_cell_vertices = cell_vertices.cols();

  // Collect up cells on common vertices

  // Reset map
  sh_vert_to_cell.clear();
  std::vector<std::int64_t> cell_set;
  for (int i = 0; i < mpi_size; ++i)
  {
    const std::vector<std::int64_t>& recv_i = recv_vertcells[i];
    for (auto q = recv_i.begin(); q != recv_i.end(); q += num_cell_vertices + 1)
    {
      const std::size_t vertex_index = *(q + 1);
      // Packing: [owner, cell_index, this_vertex, [other_vertices]]
      cell_set = {i};
      cell_set.insert(cell_set.end(), q, q + num_cell_vertices + 1);

      // Look for vertex in map, and add the attached cell
      auto it = sh_vert_to_cell.insert({vertex_index, cell_set});
      if (!it.second)
        it.first->second.insert(it.first->second.end(), cell_set.begin(),
                                cell_set.end());
    }
  }

  // Clear sending arrays
  send_vertcells = std::vector<std::vector<std::int64_t>>(mpi_size);

  // Send back out to all processes which share the same vertex
  for (const auto& p : sh_vert_to_cell)
  {
    for (auto q = p.second.begin(); q != p.second.end();
         q += (num_cell_vertices + 2))
    {
      send_vertcells[*q].insert(send_vertcells[*q].end(), p.second.begin(),
                                p.second.end());
    }
  }

  dolfin::MPI::all_to_all(mpi_comm, send_vertcells, recv_vertcells);

  // Count up new cells, assign local index, set owner and initialise
  // shared_cells

  const std::int32_t num_cells = cell_vertices.rows();
  std::int32_t count = num_cells;

  for (const auto& p : recv_vertcells)
  {
    for (auto q = p.begin(); q != p.end(); q += num_cell_vertices + 2)
    {
      const std::int64_t owner = *q;
      const std::int64_t cell_index = *(q + 1);

      auto cell_insert = cell_global_to_local.insert({cell_index, count});
      if (cell_insert.second)
      {
        shared_cells.insert({count, std::set<std::int32_t>()});
        global_cell_indices.push_back(cell_index);
        cell_partition.push_back(owner);
        ++count;
      }
    }
  }

  // Add received cells and update sharing information for cells
  cell_vertices.conservativeResize(count, num_cell_vertices);

  // Set of processes and cells sharing the same vertex
  std::set<std::int32_t> sharing_procs;
  std::vector<std::size_t> sharing_cells;

  std::size_t last_vertex = std::numeric_limits<std::size_t>::max();
  for (const auto& p : recv_vertcells)
  {
    for (auto q = p.begin(); q != p.end(); q += num_cell_vertices + 2)
    {
      const int owner = *q;
      const std::size_t cell_index = *(q + 1);
      const std::size_t shared_vertex = *(q + 2);
      const std::int32_t local_index
          = cell_global_to_local.find(cell_index)->second;

      // Add vertices to new cells
      if (local_index >= num_cells)
      {
        for (std::int32_t j = 0; j != num_cell_vertices; ++j)
          cell_vertices(local_index, j) = *(q + j + 2);
      }

      // If starting on a new shared vertex, dump old data into
      // shared_cells
      if (shared_vertex != last_vertex)
      {
        last_vertex = shared_vertex;
        for (const auto& c : sharing_cells)
        {
          auto it = shared_cells.insert({c, sharing_procs});
          if (!it.second)
            it.first->second.insert(sharing_procs.begin(), sharing_procs.end());
        }
        sharing_procs.clear();
        sharing_cells.clear();
      }

      // Don't include self in sharing processes
      if (owner != mpi_rank)
        sharing_procs.insert(owner);
      sharing_cells.push_back(local_index);
    }
  }

  // Dump data from final vertex into shared_cells
  for (const auto& c : sharing_cells)
  {
    auto it = shared_cells.insert({c, sharing_procs});
    if (!it.second)
      it.first->second.insert(sharing_procs.begin(), sharing_procs.end());
  }
}

} // namespace
//-----------------------------------------------------------------------------
// Distribute points
std::pair<common::IndexMap,
          Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
Partitioning::distribute_points(
    MPI_Comm comm,
    Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>
        points,
    const std::vector<std::int64_t>& global_point_indices)
{
  common::Timer timer("Distribute points");

  int mpi_size = MPI::size(comm);
  int mpi_rank = MPI::rank(comm);

  // Compute where (process number) the points we need are located
  std::vector<std::int64_t> ranges(mpi_size);
  MPI::all_gather(comm, (std::int64_t)points.rows(), ranges);
  for (std::size_t i = 1; i < ranges.size(); ++i)
    ranges[i] += ranges[i - 1];
  ranges.insert(ranges.begin(), 0);

  std::vector<int> send_offsets(mpi_size);
  std::vector<std::int64_t>::const_iterator it = global_point_indices.begin();
  for (int i = 0; i < mpi_size; ++i)
  {
    // Find first index on each process
    it = std::lower_bound(it, global_point_indices.end(), ranges[i]);
    send_offsets[i] = it - global_point_indices.begin();
  }
  send_offsets.push_back(global_point_indices.size());

  std::vector<int> send_sizes(mpi_size);
  for (int i = 0; i < mpi_size; ++i)
    send_sizes[i] = send_offsets[i + 1] - send_offsets[i];

  // Get data size to transfer in Alltoallv
  std::vector<int> recv_sizes(mpi_size);
  MPI_Alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1, MPI_INT,
               comm);

  std::vector<int> recv_offsets(mpi_size + 1, 0);
  std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                   recv_offsets.begin() + 1);
  std::vector<std::int64_t> recv_global_index(recv_offsets.back());

  // Transfer global indices to the processes holding the points
  MPI_Alltoallv(global_point_indices.data(), send_sizes.data(),
                send_offsets.data(), MPI::mpi_type<std::int64_t>(),
                recv_global_index.data(), recv_sizes.data(),
                recv_offsets.data(), MPI::mpi_type<std::int64_t>(), comm);

  // Determine ownership
  // Lowest rank will be the index owner (FIXME: to be revised)
  std::vector<int> index_owner(ranges[mpi_rank + 1] - ranges[mpi_rank], -1);
  std::vector<int> count(mpi_size);
  for (int i = 0; i < mpi_size; ++i)
  {
    for (int j = recv_offsets[i]; j < recv_offsets[i + 1]; ++j)
    {
      const std::int64_t gi = recv_global_index[j] - ranges[mpi_rank];
      if (index_owner[gi] < 0)
      {
        index_owner[gi] = i;
        ++count[i];
      }
    }
  }

  // Get global offsets - each process contains a portion of owned indices
  // of each other process. FIXME: optimise this section
  std::vector<int> count_remote(mpi_size);
  MPI_Alltoall(count.data(), 1, MPI_INT, count_remote.data(), 1, MPI_INT, comm);
  std::vector<int> count_sum(mpi_size + 1, 0);
  std::partial_sum(count_remote.begin(), count_remote.end(),
                   count_sum.begin() + 1);
  const std::int64_t local_size = count_sum.back();
  count_sum[0] = local_size;
  std::cout << "size on process " << mpi_rank << " = " << local_size << "\n";

  // Send offsets back to holding processes
  MPI_Alltoall(count_sum.data(), 1, MPI_INT, count_remote.data(), 1, MPI_INT,
               comm);
  if (mpi_rank == 0)
  {
    count_sum[0] = 0;
    std::partial_sum(count_remote.begin(), count_remote.end(),
                     count_sum.begin() + 1);
    std::fill(count_remote.begin(), count_remote.end(), 0);
    std::cout << "total verts = " << count_sum.back() << "!!\n";
  }
  MPI_Bcast(count_sum.data(), mpi_size, MPI_INT, 0, comm);
  std::int64_t local_offset = count_sum[mpi_rank];
  std::cout << "Offset of process " << mpi_rank << " = " << local_offset
            << "\n";
  for (int i = 0; i < mpi_size; ++i)
    count_sum[i] += count_remote[i];

  // Make new global indexing, taking ghosting into account
  std::vector<std::int64_t> new_global_index0(index_owner.size(), -1);
  // Label 'local' indices first
  for (int i = 0; i < mpi_size; ++i)
  {
    for (int j = recv_offsets[i]; j < recv_offsets[i + 1]; ++j)
    {
      const std::int64_t gi = recv_global_index[j] - ranges[mpi_rank];
      if (index_owner[gi] == i)
      {
        assert(new_global_index0[gi] == -1);
        new_global_index0[gi] = count_sum[i];
        ++count_sum[i];
      }
      else
        std::cout << "+";
    }
  }

  // Second pass, fill in return data, including ghost indices
  std::vector<std::int64_t> new_global_index(recv_global_index.size());
  for (int i = 0; i < mpi_size; ++i)
  {
    for (int j = recv_offsets[i]; j < recv_offsets[i + 1]; ++j)
    {
      const std::int64_t gi = recv_global_index[j] - ranges[mpi_rank];
      assert(new_global_index0[gi] != -1);
      new_global_index[j] = new_global_index0[gi];
    }
  }

  std::vector<std::int64_t> recv_new_global(send_offsets.back());
  MPI_Alltoallv(new_global_index.data(), recv_sizes.data(), recv_offsets.data(),
                MPI::mpi_type<std::int64_t>(), recv_new_global.data(),
                send_sizes.data(), send_offsets.data(),
                MPI::mpi_type<std::int64_t>(), comm);

  std::vector<std::int32_t> new_local;
  std::vector<std::int64_t> ghosts;
  for (std::int64_t r : recv_new_global)
  {
    std::int64_t rlocal = r - local_offset;
    if (rlocal < 0 or rlocal >= local_size)
    {
      new_local.push_back(local_size + ghosts.size());
      ghosts.push_back(r);
    }
    else
      new_local.push_back(rlocal);
  }
  std::cout << "ghosts.size=" << ghosts.size() << "\n";

  Eigen::Map<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>> garr(ghosts.data(),
                                                                 ghosts.size());

  common::IndexMap v_idx_map(comm, local_size, garr, 1);

  std::cout << mpi_rank << "] " << v_idx_map.size_local() << " "
            << v_idx_map.size_global() << " " << v_idx_map.num_ghosts() << " "
            << v_idx_map.ghost_owners() << "\n";

  // Create compound datatype of gdim*doubles (point coords)
  MPI_Datatype compound_f64;
  MPI_Type_contiguous(points.cols(), MPI_DOUBLE, &compound_f64);
  MPI_Type_commit(&compound_f64);

  // Fill in points to send back
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      send_points(recv_global_index.size(), points.cols());
  for (std::size_t i = 0; i < recv_global_index.size(); ++i)
  {
    assert(recv_global_index[i] >= ranges[mpi_rank]);
    assert(recv_global_index[i] < ranges[mpi_rank + 1]);

    int local_index = recv_global_index[i] - ranges[mpi_rank];
    send_points.row(i) = points.row(local_index);
  }

  // Send points back, matching indices in global_index_set
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      recv_points(global_point_indices.size(), points.cols());

  MPI_Alltoallv(send_points.data(), recv_sizes.data(), recv_offsets.data(),
                compound_f64, recv_points.data(), send_sizes.data(),
                send_offsets.data(), compound_f64, comm);
  timer.stop();

  // Sort points and input global_indices into new order...

  return std::pair(std::move(v_idx_map), std::move(recv_points));
}
//-----------------------------------------------------------------------------
// Compute cell partitioning from local mesh data. Returns a vector
// 'cell -> process' vector for cells, and a map 'local cell index ->
// processes' to which ghost cells must be sent
PartitionData Partitioning::partition_cells(
    const MPI_Comm& mpi_comm, int nparts, const mesh::CellType cell_type,
    const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>&
        cell_vertices,
    const mesh::Partitioner graph_partitioner)
{
  LOG(INFO) << "Compute partition of cells across processes";

  // If this process is not in the (new) communicator, it will be
  // MPI_COMM_NULL.
  if (mpi_comm != MPI_COMM_NULL)
  {
    // Compute dual graph (for this partition)
    auto [local_graph, graph_info] = graph::GraphBuilder::compute_dual_graph(
        mpi_comm, cell_vertices, cell_type);

    const std::size_t global_graph_size
        = MPI::sum(mpi_comm, local_graph.size());
    const std::size_t num_processes = MPI::size(mpi_comm);

    // Require at least two cells per processor for mesh partitioning in
    // parallel. Partitioning small graphs may lead to segfaults or MPI
    // processes with 0 cells.
    if (num_processes > 1 and global_graph_size / nparts < 2)
    {
      throw std::runtime_error("Cannot partition a graph of size "
                               + std::to_string(global_graph_size) + " into "
                               + std::to_string(nparts) + " parts.");
    }

    // Compute cell partition using partitioner from parameter system
    if (graph_partitioner == mesh::Partitioner::scotch)
    {
      graph::CSRGraph<SCOTCH_Num> csr_graph(mpi_comm, local_graph);
      std::vector<std::size_t> weights;
      const std::int32_t num_ghost_nodes = std::get<0>(graph_info);
      return PartitionData(graph::SCOTCH::partition(
          mpi_comm, (SCOTCH_Num)nparts, csr_graph, weights, num_ghost_nodes));
    }
    else if (graph_partitioner == mesh::Partitioner::parmetis)
    {
#ifdef HAS_PARMETIS
      graph::CSRGraph<idx_t> csr_graph(mpi_comm, local_graph);
      return PartitionData(
          graph::ParMETIS::partition(mpi_comm, (idx_t)nparts, csr_graph));
#else
      throw std::runtime_error("ParMETIS not available");
#endif
    }
    else if (graph_partitioner == mesh::Partitioner::kahip)
    {
#ifdef HAS_KAHIP
      graph::CSRGraph<unsigned long long> csr_graph(mpi_comm, local_graph);
      return PartitionData(
          graph::KaHIP::partition(mpi_comm, nparts, csr_graph));
#else
      throw std::runtime_error("KaHIP not available");
#endif
    }
    else
      throw std::runtime_error("Unknown graph partitioner");

    return PartitionData({}, {});
  }
  return PartitionData({}, {});
}
//-----------------------------------------------------------------------------
// Build a distributed mesh from local mesh data with a computed
// partition
mesh::Mesh Partitioning::build_from_partition(
    const MPI_Comm& comm, mesh::CellType cell_type,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& points,
    const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>&
        cell_vertices,
    const std::vector<std::int64_t>& global_cell_indices,
    const mesh::GhostMode ghost_mode, const PartitionData& cell_partition)
{
  LOG(INFO) << "Distribute mesh cells";

  common::Timer timer("Distribute mesh cells");

  // Check that we have some ghost information.
  int all_ghosts = dolfin::MPI::sum(comm, cell_partition.num_ghosts());
  if (all_ghosts == 0 and ghost_mode != mesh::GhostMode::none)
    throw std::runtime_error("Ghost cell information not available");

  // Topological dimension
  const int tdim = mesh::cell_dim(cell_type);

  // Send cells to owning process according to cell_partition, and
  // receive cells that belong to this process. Also compute auxiliary
  // data related to sharing.
  Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      new_cell_vertices;
  std::vector<std::int64_t> new_global_cell_indices;
  std::vector<int> new_cell_partition;
  std::map<std::int32_t, std::set<std::int32_t>> shared_cells;
  std::int32_t num_regular_cells;
  std::tie(new_cell_vertices, new_global_cell_indices, new_cell_partition,
           shared_cells, num_regular_cells)
      = distribute_cells(comm, cell_vertices, global_cell_indices,
                         cell_partition);

  if (ghost_mode == mesh::GhostMode::shared_vertex)
  {
    // Send/receive additional cells defined by connectivity to the shared
    // vertices.
    distribute_cell_layer(comm, num_regular_cells, shared_cells,
                          new_cell_vertices, new_global_cell_indices,
                          new_cell_partition);
  }
  else if (ghost_mode == mesh::GhostMode::none)
  {
    // Resize to remove all ghost cells
    new_cell_partition.resize(num_regular_cells);
    new_global_cell_indices.resize(num_regular_cells);
    new_cell_vertices.conservativeResize(num_regular_cells, Eigen::NoChange);
    shared_cells.clear();
  }

  // if (parameter::parameters["reorder_cells_gps"])
  // {
  //   // Allocate objects to hold re-ordering
  //   std::map<std::int32_t, std::set<std::int32_t>> reordered_shared_cells;
  //   EigenRowArrayXXi64 reordered_cell_vertices;
  //   std::vector<std::int64_t> reordered_global_cell_indices;

  //   // Re-order cells
  //   std::tie(reordered_shared_cells, reordered_cell_vertices,
  //            reordered_global_cell_indices)
  //       = reorder_cells_gps(comm, num_regular_cells, *cell_type,
  //       shared_cells,
  //                           new_cell_vertices, new_global_cell_indices);

  //   // Update to re-ordered indices
  //   std::swap(shared_cells, reordered_shared_cells);
  //   new_cell_vertices = reordered_cell_vertices;
  //   std::swap(new_global_cell_indices, reordered_global_cell_indices);
  // }

  timer.stop();

  // Build mesh from points and distributed cells
  const std::int32_t num_ghosts = new_cell_vertices.rows() - num_regular_cells;

  mesh::Mesh mesh(comm, cell_type, points, new_cell_vertices,
                  new_global_cell_indices, ghost_mode, num_ghosts);

  if (ghost_mode == mesh::GhostMode::none)
    return mesh;

  // Copy cell ownership (only needed for ghost cells)
  std::vector<std::int32_t> cell_owner(
      new_cell_partition.begin() + num_regular_cells, new_cell_partition.end());

  // Assign map of shared cells (only needed for ghost cells)
  mesh.topology().set_entity_owner(tdim, cell_owner);
  mesh.topology().set_shared_entities(tdim, shared_cells);
  DistributedMeshTools::init_facet_cell_connections(mesh);

  return mesh;
}
//-----------------------------------------------------------------------------
mesh::Mesh Partitioning::build_distributed_mesh(
    const MPI_Comm& comm, mesh::CellType cell_type,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& points,
    const Eigen::Ref<const Eigen::Array<
        std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& cells,
    const std::vector<std::int64_t>& global_cell_indices,
    const mesh::GhostMode ghost_mode, const mesh::Partitioner graph_partitioner)
{

  // By default all processes are used to partition the mesh
  // nparts = MPI size
  const int nparts = dolfin::MPI::size(comm);

  // Compute the cell partition
  PartitionData cell_partition
      = partition_cells(comm, nparts, cell_type, cells, graph_partitioner);

  // Build mesh from local mesh data and provided cell partition
  mesh::Mesh mesh = Partitioning::build_from_partition(
      comm, cell_type, points, cells, global_cell_indices, ghost_mode,
      cell_partition);

  // Initialise number of globally connected cells to each facet. This
  // is necessary to distinguish between facets on an exterior boundary
  // and facets on a partition boundary (see
  // https://bugs.launchpad.net/dolfin/+bug/733834).

  DistributedMeshTools::init_facet_cell_connections(mesh);

  return mesh;
}
//-----------------------------------------------------------------------------
std::tuple<std::int64_t, std::vector<std::int64_t>,
           std::shared_ptr<common::IndexMap>>
Partitioning::build_global_vertex_indices(
    MPI_Comm mpi_comm, const std::array<std::int32_t, 4>& num_vertices,
    const std::vector<std::int64_t>& global_point_indices,
    const std::map<std::int32_t, std::set<std::int32_t>>& shared_points)
{
  // Find out how many vertices are locally 'owned' and number them
  std::vector<std::int64_t> global_vertex_indices(num_vertices[2]);
  const std::int32_t mpi_rank = dolfin::MPI::rank(mpi_comm);
  const std::int32_t mpi_size = dolfin::MPI::size(mpi_comm);
  std::vector<std::vector<std::int64_t>> send_data(mpi_size);
  std::vector<std::int64_t> recv_data(mpi_size);

  // Local
  std::int64_t v = 0;
  for (std::int32_t i = 0; i < num_vertices[0]; ++i)
  {
    global_vertex_indices[i] = v;
    ++v;
  }

  // Shared and owned (ghost on other processes)
  for (std::int32_t i = num_vertices[0]; i < num_vertices[1]; ++i)
  {
    const auto it = shared_points.find(i);
    assert(it != shared_points.end());
    assert(mpi_rank < *it->second.begin());

    global_vertex_indices[i] = v;
    for (auto p : it->second)
    {
      send_data[p].push_back(global_point_indices[i]);
      send_data[p].push_back(v);
    }
    ++v;
  }

  // Now have numbered all vertices locally so can get global size and
  // local offset
  std::int64_t num_vertices_global = dolfin::MPI::sum(mpi_comm, v);
  std::int64_t offset = dolfin::MPI::global_offset(mpi_comm, v, true);

  // Add offset to send_data
  for (auto& p : send_data)
    for (auto q = p.begin(); q != p.end(); q += 2)
      *(q + 1) += offset;

  // Receive indices of vertices owned elsewhere into map
  dolfin::MPI::all_to_all(mpi_comm, send_data, recv_data);
  std::map<std::int64_t, std::int64_t> global_point_to_vertex;
  for (auto p = recv_data.begin(); p != recv_data.end(); p += 2)
  {
    auto it = global_point_to_vertex.insert({*p, *(p + 1)});
    assert(it.second);
  }

  // Adjust global_vertex_indices either by adding offset or inserting
  // remote index
  for (std::int32_t i = 0; i < num_vertices[1]; ++i)
    global_vertex_indices[i] += offset;
  for (std::int32_t i = num_vertices[1]; i < num_vertices[2]; ++i)
  {
    const auto it = global_point_to_vertex.find(global_point_indices[i]);
    assert(it != global_point_to_vertex.end());
    global_vertex_indices[i] = it->second;
  }

  Eigen::Map<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>> ghosts(
      global_vertex_indices.data() + num_vertices[1],
      num_vertices[2] - num_vertices[1]);

  // Test creating IndexMap
  auto vertex_index_map = std::make_shared<common::IndexMap>(
      mpi_comm, num_vertices[1], ghosts, 1);

  return {num_vertices_global, std::move(global_vertex_indices),
          vertex_index_map};
}
//-----------------------------------------------------------------------------
std::map<std::int64_t, std::vector<int>> Partitioning::compute_halo_cells(
    MPI_Comm mpi_comm, std::vector<int> part, const mesh::CellType cell_type,
    const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>&
        cell_vertices)
{
  // Compute dual graph (for this partition)
  std::vector<std::vector<std::size_t>> local_graph;
  std::tie(local_graph, std::ignore) = graph::GraphBuilder::compute_dual_graph(
      mpi_comm, cell_vertices, cell_type);

  graph::CSRGraph<std::int64_t> csr_graph(mpi_comm, local_graph);

  std::map<std::int64_t, std::vector<int>> ghost_procs;

  // Work out halo cells for current division of dual graph
  const std::vector<std::int64_t>& elmdist = csr_graph.node_distribution();
  const std::vector<std::int64_t>& xadj = csr_graph.nodes();
  const std::vector<std::int64_t>& adjncy = csr_graph.edges();
  const std::int32_t num_processes = dolfin::MPI::size(mpi_comm);
  const std::int32_t process_number = dolfin::MPI::rank(mpi_comm);
  const std::int32_t elm_begin = elmdist[process_number];
  const std::int32_t elm_end = elmdist[process_number + 1];
  const std::int32_t ncells = elm_end - elm_begin;

  std::map<std::int32_t, std::set<std::int32_t>> halo_cell_to_remotes;

  for (int i = 0; i < ncells; i++)
  {
    for (auto other_cell : csr_graph[i])
    {
      if (other_cell < elm_begin || other_cell >= elm_end)
      {
        const int remote
            = std::upper_bound(elmdist.begin(), elmdist.end(), other_cell)
              - elmdist.begin() - 1;
        assert(remote < num_processes);
        if (halo_cell_to_remotes.find(i) == halo_cell_to_remotes.end())
          halo_cell_to_remotes[i] = std::set<std::int32_t>();
        halo_cell_to_remotes[i].insert(remote);
      }
    }
  }

  // Do halo exchange of cell partition data
  std::vector<std::vector<std::int64_t>> send_cell_partition(num_processes);
  std::vector<std::int64_t> recv_cell_partition;

  for (const auto& hcell : halo_cell_to_remotes)
  {
    for (auto proc : hcell.second)
    {
      assert(proc < num_processes);

      // global cell number
      send_cell_partition[proc].push_back(hcell.first + elm_begin);

      // partitioning
      send_cell_partition[proc].push_back(part[hcell.first]);
    }
  }

  // Actual halo exchange
  dolfin::MPI::all_to_all(mpi_comm, send_cell_partition, recv_cell_partition);

  // Construct a map from all currently foreign cells to their new
  // partition number
  std::map<std::int64_t, std::int32_t> cell_ownership;
  for (auto p = recv_cell_partition.begin(); p != recv_cell_partition.end();
       p += 2)
  {
    cell_ownership[*p] = *(p + 1);
  }

  // Generate mapping for where new boundary cells need to be sent
  for (std::int32_t i = 0; i < ncells; i++)
  {
    const std::size_t proc_this = part[i];
    for (std::int32_t j = xadj[i]; j < xadj[i + 1]; ++j)
    {
      const std::int32_t other_cell = adjncy[j];
      std::size_t proc_other;

      if (other_cell < elm_begin || other_cell >= elm_end)
      { // remote cell - should be in map
        const auto find_other_proc = cell_ownership.find(other_cell);
        assert(find_other_proc != cell_ownership.end());
        proc_other = find_other_proc->second;
      }
      else
        proc_other = part[other_cell - elm_begin];

      if (proc_this != proc_other)
      {
        auto map_it = ghost_procs.find(i);
        if (map_it == ghost_procs.end())
        {
          std::vector<std::int32_t> sharing_processes;
          sharing_processes.push_back(proc_this);
          sharing_processes.push_back(proc_other);
          ghost_procs.insert({i, sharing_processes});
        }
        else
        {
          // Add to vector if not already there
          auto it = std::find(map_it->second.begin(), map_it->second.end(),
                              proc_other);
          if (it == map_it->second.end())
            map_it->second.push_back(proc_other);
        }
      }
    }
  }

  return ghost_procs;
}
