// Copyright (C) 2007-2022 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MPI.h"
#include <dolfinx/common/log.h>
#include <iostream>

//-----------------------------------------------------------------------------
dolfinx::MPI::Comm::Comm(MPI_Comm comm, bool duplicate)
{
  // Duplicate communicator
  if (duplicate and comm != MPI_COMM_NULL)
  {
    int err = MPI_Comm_dup(comm, &_comm);
    dolfinx::MPI::check_error(comm, err);
  }
  else
    _comm = comm;
}
//-----------------------------------------------------------------------------
dolfinx::MPI::Comm::Comm(const Comm& comm) noexcept
    : dolfinx::MPI::Comm::Comm(comm._comm, true)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfinx::MPI::Comm::Comm(Comm&& comm) noexcept
{
  this->_comm = comm._comm;
  comm._comm = MPI_COMM_NULL;
}
//-----------------------------------------------------------------------------
dolfinx::MPI::Comm::~Comm()
{
  // Free the comm
  if (_comm != MPI_COMM_NULL)
  {
    int err = MPI_Comm_free(&_comm);
    dolfinx::MPI::check_error(_comm, err);
  }
}
//-----------------------------------------------------------------------------
dolfinx::MPI::Comm&
dolfinx::MPI::Comm::operator=(dolfinx::MPI::Comm&& comm) noexcept
{
  // Free the currently held comm
  if (this->_comm != MPI_COMM_NULL)
  {
    int err = MPI_Comm_free(&this->_comm);
    dolfinx::MPI::check_error(this->_comm, err);
  }

  // Move comm from other object
  this->_comm = comm._comm;
  comm._comm = MPI_COMM_NULL;
  return *this;
}
//-----------------------------------------------------------------------------
MPI_Comm dolfinx::MPI::Comm::comm() const noexcept { return _comm; }
//-----------------------------------------------------------------------------
int dolfinx::MPI::rank(const MPI_Comm comm)
{
  int rank;
  int err = MPI_Comm_rank(comm, &rank);
  dolfinx::MPI::check_error(comm, err);
  return rank;
}
//-----------------------------------------------------------------------------
int dolfinx::MPI::size(const MPI_Comm comm)
{
  int size;
  int err = MPI_Comm_size(comm, &size);
  dolfinx::MPI::check_error(comm, err);
  return size;
}
//-----------------------------------------------------------------------------
void dolfinx::MPI::check_error(MPI_Comm comm, int code)
{
  if (code != MPI_SUCCESS)
  {
    int len = MPI_MAX_ERROR_STRING;
    std::string error_string(MPI_MAX_ERROR_STRING, ' ');
    MPI_Error_string(code, error_string.data(), &len);
    error_string.resize(len);
    std::cerr << error_string << std::endl;
    MPI_Abort(comm, code);
    std::abort();
  }
}
//-----------------------------------------------------------------------------
std::vector<int>
dolfinx::MPI::compute_graph_edges_pcx(MPI_Comm comm, std::span<const int> edges)
{
  LOG(INFO)
      << "Computing communication graph edges (using PCX algorithm). Number "
         "of input edges: "
      << edges.size();

  // Build array with '0' for no outedge and '1' for an outedge for each
  // rank
  const int size = dolfinx::MPI::size(comm);
  std::vector<int> edge_count_send(size, 0);
  for (auto e : edges)
    edge_count_send[e] = 1;

  // Determine how many in-edges this rank has
  std::vector<int> recvcounts(size, 1);
  int in_edges = 0;
  MPI_Request request_scatter;
  int err = MPI_Ireduce_scatter(edge_count_send.data(), &in_edges,
                                recvcounts.data(), MPI_INT, MPI_SUM, comm,
                                &request_scatter);
  dolfinx::MPI::check_error(comm, err);

  std::vector<MPI_Request> send_requests(edges.size());
  std::byte send_buffer;
  for (std::size_t e = 0; e < edges.size(); ++e)
  {
    int err = MPI_Isend(&send_buffer, 1, MPI_BYTE, edges[e],
                        static_cast<int>(tag::consensus_pcx), comm,
                        &send_requests[e]);
    dolfinx::MPI::check_error(comm, err);
  }

  // Probe for incoming messages and store incoming rank
  err = MPI_Wait(&request_scatter, MPI_STATUS_IGNORE);
  dolfinx::MPI::check_error(comm, err);
  std::vector<int> other_ranks;
  while (in_edges > 0)
  {
    // Check for message
    int request_pending;
    MPI_Status status;
    int err = MPI_Iprobe(MPI_ANY_SOURCE, static_cast<int>(tag::consensus_pcx),
                         comm, &request_pending, &status);
    dolfinx::MPI::check_error(comm, err);
    if (request_pending)
    {
      // Receive message and store rank
      int other_rank = status.MPI_SOURCE;
      std::byte buffer_recv;
      int err = MPI_Recv(&buffer_recv, 1, MPI_BYTE, other_rank,
                         static_cast<int>(tag::consensus_pcx), comm,
                         MPI_STATUS_IGNORE);
      dolfinx::MPI::check_error(comm, err);
      other_ranks.push_back(other_rank);
      --in_edges;
    }
  }

  LOG(INFO) << "Finished graph edge discovery using PCX algorithm. Number "
               "of discovered edges "
            << other_ranks.size();

  return other_ranks;
}
//-----------------------------------------------------------------------------
std::pair<std::vector<int>, std::vector<int>>
transpose_src_dest(std::vector<int>& dests, std::vector<int>& dest_offsets)
{
  // List of destinations, ordered by source
  // to be transposed into a list of sources, ordered by destination

  std::vector<int> source_count(dest_offsets.size() - 1);
  for (int d : dests)
    ++source_count[d];
  std::vector<int> source_offsets = {0};
  for (int c : source_count)
    source_offsets.push_back(source_offsets.back() + c);
  std::vector<int> sources(source_offsets.back());
  for (std::size_t i = 0; i < dest_offsets.size() - 1; ++i)
  {
    for (int d = dest_offsets[i]; d < dest_offsets[i + 1]; ++d)
    {
      int dest = dests[d];
      sources[source_offsets[dest]] = i;
      source_offsets[dest]++;
    }
  }
  // Reset offsets
  source_offsets = {0};
  for (int c : source_count)
    source_offsets.push_back(source_offsets.back() + c);

  return {std::move(sources), std::move(source_offsets)};
}
//-----------------------------------------------------------------------------
std::vector<int>
dolfinx::MPI::compute_graph_edges_nbx(MPI_Comm comm, std::span<const int> edges)
{
  int rank = dolfinx::MPI::rank(comm);
  int size = dolfinx::MPI::size(comm);
  int num_edges = edges.size();
  std::vector<int> edges_count_node;
  edges_count_node.reserve(1);
  if (rank == 0)
    edges_count_node.resize(size);
  MPI_Gather(&num_edges, 1, MPI_INT, edges_count_node.data(), 1, MPI_INT, 0,
             comm);
  std::vector<int> edges_offset_node = {0};
  for (int n : edges_count_node)
    edges_offset_node.push_back(edges_offset_node.back() + n);

  std::vector<int> edges_node;
  edges_node.reserve(1);
  edges_node.resize(edges_offset_node.back());
  MPI_Gatherv(edges.data(), edges.size(), MPI_INT, edges_node.data(),
              edges_count_node.data(), edges_offset_node.data(), MPI_INT, 0,
              comm);

  auto [reply_node, reply_offset_node]
      = transpose_src_dest(edges_node, edges_offset_node);

  std::vector<int> reply_count(reply_offset_node.size() - 1);
  for (std::size_t i = 0; i < reply_count.size(); ++i)
    reply_count[i] = reply_offset_node[i + 1] - reply_offset_node[i];
  MPI_Scatter(reply_count.data(), 1, MPI_INT, &num_edges, 1, MPI_INT, 0, comm);
  std::vector<int> out_edges(num_edges);
  MPI_Scatterv(reply_node.data(), reply_count.data(), reply_offset_node.data(),
               MPI_INT, out_edges.data(), num_edges, MPI_INT, 0, comm);

  return out_edges;
}
//-----------------------------------------------------------------------------
// std::vector<int>
// dolfinx::MPI::compute_graph_edges_nbx(MPI_Comm comm, std::span<const int>
// edges)
// {
//   LOG(INFO)
//       << "Computing communication graph edges (using NBX algorithm). Number "
//          "of input edges: "
//       << edges.size();

//   // Create a sub-communicator on node
//   MPI_Comm shm_comm;
//   MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
//   &shm_comm); int local_rank = dolfinx::MPI::rank(shm_comm); int local_size =
//   dolfinx::MPI::size(shm_comm); LOG(INFO) << "Created shm_comm of size " <<
//   local_size;

//   // Create a comm across nodes, using rank 0 of the local comm on each node
//   MPI_Comm sub_comm;
//   int color = (local_rank == 0) ? 0 : MPI_UNDEFINED;
//   MPI_Comm_split(comm, color, 0, &sub_comm);

//   LOG(INFO) << "Created sub_comm of size " << dolfinx::MPI::size(sub_comm);

//   // Collect all local input edges on rank 0 of shm_comm
//   int num_edges = edges.size();
//   std::vector<int> edges_count_node;
//   edges_count_node.reserve(1);
//   if (local_rank == 0)
//     edges_count_node.resize(local_size);
//   MPI_Gather(&num_edges, 1, MPI_INT, edges_count_node.data(), 1, MPI_INT, 0,
//              shm_comm);
//   std::vector<int> edges_offset_node = {0};
//   for (int n : edges_count_node)
//     edges_offset_node.push_back(edges_offset_node.back() + n);

//   std::vector<int> edges_node;
//   edges_node.reserve(1);
//   edges_node.resize(edges_offset_node.back());
//   MPI_Gatherv(edges.data(), edges.size(), MPI_INT, edges_node.data(),
//               edges_count_node.data(), edges_offset_node.data(), MPI_INT, 0,
//               shm_comm);

//   // Vector to hold data from other ranks
//   std::vector<std::array<int, 2>> other_rank_data;

//   if (local_rank == 0)
//   {
//     // Start non-blocking synchronised send on sub_comm
//     int rank = dolfinx::MPI::rank(comm);
//     std::vector<MPI_Request> send_requests(edges_node.size());
//     std::vector<int> src_dest;
//     for (int i = 0; i < local_size; ++i)
//     {
//       for (int e = edges_offset_node[i]; e < edges_offset_node[i + 1]; ++e)
//       {
//         src_dest.push_back(edges_node[e]);
//         src_dest.push_back(rank + i);
//       }
//     }

//     for (std::size_t i = 0; i < src_dest.size() / 2; ++i)
//     {
//       int dest = src_dest[i * 2] / local_size;
//       int err = MPI_Issend(src_dest.data() + 2 * i, 2, MPI_INT, dest,
//                            static_cast<int>(tag::consensus_pex), sub_comm,
//                            &send_requests[i]);
//       dolfinx::MPI::check_error(sub_comm, err);
//     }

//     // Start sending/receiving
//     MPI_Request barrier_request;
//     bool comm_complete = false;
//     bool barrier_active = false;
//     while (!comm_complete)
//     {
//       // Check for message
//       int request_pending;
//       MPI_Status status;
//       int err = MPI_Iprobe(MPI_ANY_SOURCE,
//       static_cast<int>(tag::consensus_pex),
//                            sub_comm, &request_pending, &status);
//       dolfinx::MPI::check_error(sub_comm, err);

//       // Check if message is waiting to be processed
//       if (request_pending)
//       {
//         // Receive it
//         int other_rank = status.MPI_SOURCE;
//         std::array<int, 2> buffer_recv;
//         int err = MPI_Recv(buffer_recv.data(), 2, MPI_INT, other_rank,
//                            static_cast<int>(tag::consensus_pex), sub_comm,
//                            MPI_STATUS_IGNORE);
//         dolfinx::MPI::check_error(sub_comm, err);
//         other_rank_data.push_back(buffer_recv);
//       }

//       if (barrier_active)
//       {
//         // Check for barrier completion
//         int flag = 0;
//         int err = MPI_Test(&barrier_request, &flag, MPI_STATUS_IGNORE);
//         dolfinx::MPI::check_error(sub_comm, err);
//         if (flag)
//           comm_complete = true;
//       }
//       else
//       {
//         // Check if all sends have completed
//         int flag = 0;
//         int err = MPI_Testall(send_requests.size(), send_requests.data(),
//         &flag,
//                               MPI_STATUSES_IGNORE);
//         dolfinx::MPI::check_error(sub_comm, err);
//         if (flag)
//         {
//           // All sends have completed, start non-blocking barrier
//           int err = MPI_Ibarrier(sub_comm, &barrier_request);
//           dolfinx::MPI::check_error(sub_comm, err);
//           LOG(INFO) << "NBX activating barrier";
//           barrier_active = true;
//         }
//       }
//     }
//     MPI_Comm_free(&sub_comm);
//   }

//   // Distribute back to all processes on this node
//   std::vector<int> local_count(local_size, 0);
//   for (std::size_t i = 0; i < other_rank_data.size(); ++i)
//   {
//     int local_proc = other_rank_data[i][0] % local_size;
//     other_rank_data[i][0] = local_proc;
//     local_count[local_proc]++;
//   }
//   std::sort(other_rank_data.begin(), other_rank_data.end());
//   std::vector<int> other_rank_extracted;
//   for (auto q : other_rank_data)
//     other_rank_extracted.push_back(q[1]);
//   std::vector<int> local_offset = {0};
//   for (int q : local_count)
//     local_offset.push_back(local_offset.back() + q);

//   int num_other_ranks;
//   MPI_Scatter(local_count.data(), 1, MPI_INT, &num_other_ranks, 1, MPI_INT,
//   0,
//               shm_comm);
//   std::vector<int> other_ranks(num_other_ranks);
//   MPI_Scatterv(other_rank_extracted.data(), local_count.data(),
//                local_offset.data(), MPI_INT, other_ranks.data(),
//                other_ranks.size(), MPI_INT, 0, shm_comm);

//   MPI_Comm_free(&shm_comm);

//   LOG(INFO) << "Finished graph edge discovery using NBX algorithm. Number "
//                "of discovered edges "
//             << other_ranks.size();

//   return other_ranks;
// }
//-----------------------------------------------------------------------------
