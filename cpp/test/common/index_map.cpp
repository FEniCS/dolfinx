// Copyright (C) 2018 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch2/catch.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Scatterer.h>
#include <numeric>
#include <set>
#include <vector>

using namespace dolfinx;

namespace
{
void test_scatter_fwd(int n)
{
  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  const int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  const int size_local = 100;

  // Create some ghost entries on next process
  int num_ghosts = (mpi_size - 1) * 3;
  std::vector<std::int64_t> ghosts(num_ghosts);
  for (int i = 0; i < num_ghosts; ++i)
    ghosts[i] = (mpi_rank + 1) % mpi_size * size_local + i;

  std::vector<int> global_ghost_owner(ghosts.size(), (mpi_rank + 1) % mpi_size);

  // Create an IndexMap
  const common::IndexMap idx_map(MPI_COMM_WORLD, size_local, ghosts,
                                 global_ghost_owner);
  common::Scatterer sct(idx_map, n);

  // Create some data to scatter
  const std::int64_t val = 11;
  std::vector<std::int64_t> data_local(n * size_local, val * mpi_rank);
  std::vector<std::int64_t> data_ghost(n * num_ghosts, -1);

  // Scatter values to ghost and check value is correctly received
  sct.scatter_fwd<std::int64_t>(data_local, data_ghost);
  CHECK((int)data_ghost.size() == n * num_ghosts);
  CHECK(std::all_of(data_ghost.begin(), data_ghost.end(),
                    [=](auto i)
                    { return i == val * ((mpi_rank + 1) % mpi_size); }));

  std::vector<MPI_Request> requests
      = sct.create_request_vector(decltype(sct)::type::p2p);

  std::fill(data_ghost.begin(), data_ghost.end(), 0);
  sct.scatter_fwd_begin<std::int64_t>(data_local, data_ghost, requests,
                                      decltype(sct)::type::p2p);
  sct.scatter_fwd_end(requests);

  CHECK(std::all_of(data_ghost.begin(), data_ghost.end(),
                    [=](auto i)
                    { return i == val * ((mpi_rank + 1) % mpi_size); }));
}

void test_scatter_rev()
{
  // Block size
  auto n = GENERATE(1, 5, 10);

  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  const int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  const int size_local = 100;

  // Create some ghost entries on next process
  const int num_ghosts = (mpi_size - 1) * 3;
  std::vector<std::int64_t> ghosts(num_ghosts);
  for (int i = 0; i < num_ghosts; ++i)
    ghosts[i] = (mpi_rank + 1) % mpi_size * size_local + i;

  std::vector<int> global_ghost_owner(ghosts.size(), (mpi_rank + 1) % mpi_size);

  // Create an IndexMap
  const common::IndexMap idx_map(MPI_COMM_WORLD, size_local, ghosts,
                                 global_ghost_owner);
  common::Scatterer sct(idx_map, n);

  // Create some data, setting ghost values
  std::int64_t value = 15;
  std::vector<std::int64_t> data_local(n * size_local, 0);
  std::vector<std::int64_t> data_ghost(n * num_ghosts, value);
  sct.scatter_rev(std::span<std::int64_t>(data_local),
                  std::span<const std::int64_t>(data_ghost),
                  std::plus<std::int64_t>());

  std::int64_t sum;
  CHECK((int)data_local.size() == n * size_local);
  sum = std::reduce(data_local.begin(), data_local.end(), 0);
  CHECK(sum == n * value * num_ghosts);

  sct.scatter_rev(std::span<std::int64_t>(data_local),
                  std::span<const std::int64_t>(data_ghost),
                  [](auto /*a*/, auto b) { return b; });

  sum = std::reduce(data_local.begin(), data_local.end(), 0);
  CHECK(sum == n * value * num_ghosts);

  int num_requests = idx_map.dest().size() + idx_map.src().size();
  std::vector<MPI_Request> requests(num_requests, MPI_REQUEST_NULL);
  std::vector<std::int64_t> local_buffer(sct.local_buffer_size(), 0);
  std::vector<std::int64_t> remote_buffer(sct.remote_buffer_size(), 0);
  auto pack_fn = [](const auto& in, const auto& idx, auto& out)
  {
    for (std::size_t i = 0; i < idx.size(); ++i)
      out[i] = in[idx[i]];
  };
  auto unpack_fn = [](const auto& in, const auto& idx, auto& out, auto op)
  {
    for (std::size_t i = 0; i < idx.size(); ++i)
      out[idx[i]] = op(out[idx[i]], in[i]);
  };
  sct.scatter_rev_begin<std::int64_t>(data_ghost, remote_buffer, local_buffer,
                                      pack_fn, requests,
                                      decltype(sct)::type::p2p);
  //
  sct.scatter_rev_end<std::int64_t>(local_buffer, data_local, unpack_fn,
                                    std::plus<std::int64_t>(), requests);

  sum = std::reduce(data_local.begin(), data_local.end(), 0);
  CHECK(sum == 2 * n * value * num_ghosts);
}

void test_consensus_exchange()
{
  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  const int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  const int size_local = 100;

  // Create some ghost entries on next process
  const int num_ghosts = (mpi_size - 1) * 3;
  std::vector<std::int64_t> ghosts(num_ghosts);
  for (int i = 0; i < num_ghosts; ++i)
    ghosts[i] = (mpi_rank + 1) % mpi_size * size_local + i;

  std::vector<int> global_ghost_owner(ghosts.size(), (mpi_rank + 1) % mpi_size);

  // Create an IndexMap
  std::vector<int> src_ranks = global_ghost_owner;
  std::sort(src_ranks.begin(), src_ranks.end());
  src_ranks.erase(std::unique(src_ranks.begin(), src_ranks.end()),
                  src_ranks.end());

  auto dest_ranks0
      = dolfinx::MPI::compute_graph_edges_nbx(MPI_COMM_WORLD, src_ranks);
  auto dest_ranks1
      = dolfinx::MPI::compute_graph_edges_pcx(MPI_COMM_WORLD, src_ranks);
  std::sort(dest_ranks0.begin(), dest_ranks0.end());
  std::sort(dest_ranks1.begin(), dest_ranks1.end());

  CHECK(dest_ranks0 == dest_ranks1);
}
} // namespace

TEST_CASE("Scatter forward using IndexMap", "[index_map_scatter_fwd]")
{
  auto n = GENERATE(1, 5, 10);
  CHECK_NOTHROW(test_scatter_fwd(n));
}

TEST_CASE("Scatter reverse using IndexMap", "[index_map_scatter_rev]")
{
  CHECK_NOTHROW(test_scatter_rev());
}

TEST_CASE("Communication graph edges via consensus exchange",
          "[consensus_exchange]")
{
  CHECK_NOTHROW(test_consensus_exchange());
}
