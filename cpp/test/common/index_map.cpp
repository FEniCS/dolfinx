// Copyright (C) 2018 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Scatterer.h>
#include <dolfinx/common/utils.h>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>

using namespace dolfinx;

namespace
{
common::IndexMap create_index_map(MPI_Comm comm, int size_local, int num_ghosts)
{
  const int mpi_size = dolfinx::MPI::size(comm);
  const int mpi_rank = dolfinx::MPI::rank(comm);

  // Create some ghost entries on next process
  std::vector<std::int64_t> ghosts(num_ghosts);
  for (int i = 0; i < num_ghosts; ++i)
    ghosts[i] = (mpi_rank + 1) % mpi_size * size_local + i;

  std::vector<int> global_ghost_owner(ghosts.size(), (mpi_rank + 1) % mpi_size);

  // Create an IndexMap
  return common::IndexMap(MPI_COMM_WORLD, size_local, ghosts,
                          global_ghost_owner);
}

void test_scatter_fwd(int n)
{
  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  const int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  constexpr int size_local = 100;

  // Create an IndexMap
  const common::IndexMap idx_map
      = create_index_map(MPI_COMM_WORLD, size_local, (mpi_size - 1) * 3);
  std::int32_t num_ghosts = idx_map.num_ghosts();
  common::Scatterer sct(idx_map, n);

  // Create some data to scatter
  const std::int64_t val = 11;
  std::vector<std::int64_t> data_local(n * size_local, val * mpi_rank);
  std::vector<std::int64_t> data_ghost(n * num_ghosts, -1);

  // Scatter values to ghost and check value is correctly received
  {
    std::vector<std::int64_t> send_buffer(sct.local_indices().size());
    {
      auto& idx = sct.local_indices();
      for (std::size_t i = 0; i < idx.size(); ++i)
        send_buffer[i] = data_local[idx[i]];
    }
    std::vector<std::int64_t> recv_buffer(sct.remote_indices().size());
    MPI_Request request = MPI_REQUEST_NULL;
    sct.scatter_fwd_begin(send_buffer.data(), recv_buffer.data(), request);
    sct.scatter_end(request);
    {
      auto& idx = sct.remote_indices();
      for (std::size_t i = 0; i < idx.size(); ++i)
        data_ghost[idx[i]] = recv_buffer[i];
    }
    CHECK((int)data_ghost.size() == n * num_ghosts);
    CHECK(std::ranges::all_of(
        data_ghost,
        [=](auto i) { return i == val * ((mpi_rank + 1) % mpi_size); }));
  }

  {
    std::vector<MPI_Request> requests(sct.num_p2p_requests(), MPI_REQUEST_NULL);
    std::ranges::fill(data_ghost, 0);
    std::vector<std::int64_t> send_buffer(sct.local_indices().size());
    {
      auto& idx = sct.local_indices();
      for (std::size_t i = 0; i < idx.size(); ++i)
        send_buffer[i] = data_local[idx[i]];
    }
    std::vector<std::int64_t> recv_buffer(sct.remote_indices().size());
    sct.scatter_fwd_begin(send_buffer.data(), recv_buffer.data(), requests);
    sct.scatter_end(requests);
    {
      auto& idx = sct.remote_indices();
      for (std::size_t i = 0; i < idx.size(); ++i)
        data_ghost[idx[i]] = recv_buffer[i];
    }
    CHECK(std::ranges::all_of(
        data_ghost, [val, mpi_rank, mpi_size](auto i)
        { return i == val * ((mpi_rank + 1) % mpi_size); }));
  }
}

void test_scatter_rev()
{
  // Block size
  auto n = GENERATE(1, 5, 10);

  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  constexpr int size_local = 100;

  // Create an IndexMap
  const common::IndexMap idx_map
      = create_index_map(MPI_COMM_WORLD, size_local, (mpi_size - 1) * 3);
  std::int32_t num_ghosts = idx_map.num_ghosts();

  common::Scatterer sct(idx_map, n);

  auto pack_fn = [](auto&& in, auto&& idx, auto&& out)
  {
    for (std::size_t i = 0; i < idx.size(); ++i)
      out[i] = in[idx[i]];
  };
  auto unpack_fn = [](auto&& in, auto&& idx, auto&& out, auto op)
  {
    for (std::size_t i = 0; i < idx.size(); ++i)
      out[idx[i]] = op(out[idx[i]], in[i]);
  };

  // Create some data, setting ghost values
  std::int64_t value = 15;
  std::vector<std::int64_t> data_local(n * size_local, 0);
  std::vector<std::int64_t> data_ghost(n * num_ghosts, value);
  {
    std::vector<MPI_Request> requests(1, MPI_REQUEST_NULL);
    std::vector<std::int64_t> remote_buffer(sct.remote_indices().size(), 0);
    std::vector<std::int64_t> send_buffer(sct.local_indices().size(), 0);
    pack_fn(data_ghost, sct.remote_indices(), send_buffer);
    std::vector<std::int64_t> recv_buffer(sct.remote_indices().size(), 0);
    sct.scatter_rev_begin(send_buffer.data(), recv_buffer.data(), requests);
    sct.scatter_end(requests);
    unpack_fn(recv_buffer, sct.local_indices(), data_local, std::plus<>{});

    std::int64_t sum;
    CHECK((int)data_local.size() == n * size_local);
    sum = std::reduce(data_local.begin(), data_local.end(), 0);
    CHECK(sum == n * value * num_ghosts);
  }

  {
    int num_requests = idx_map.dest().size() + idx_map.src().size();
    std::vector<MPI_Request> requests(num_requests, MPI_REQUEST_NULL);
    std::vector<std::int64_t> remote_buffer(sct.remote_indices().size(), 0);

    std::vector<std::int64_t> send_buffer(sct.local_indices().size(), 0);
    pack_fn(data_ghost, sct.remote_indices(), send_buffer);
    std::vector<std::int64_t> recv_buffer(sct.remote_indices().size(), 0);
    sct.scatter_rev_begin(send_buffer.data(), recv_buffer.data(), requests);
    sct.scatter_end(requests);
    unpack_fn(recv_buffer, sct.local_indices(), data_local, std::plus<>{});

    std::int64_t sum = std::reduce(data_local.begin(), data_local.end(), 0);
    CHECK(sum == 2 * n * value * num_ghosts);
  }
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
  std::ranges::sort(src_ranks);
  auto [unique_end, range_end] = std::ranges::unique(src_ranks);
  src_ranks.erase(unique_end, range_end);

  auto dest_ranks0
      = dolfinx::MPI::compute_graph_edges_nbx(MPI_COMM_WORLD, src_ranks);
  auto dest_ranks1
      = dolfinx::MPI::compute_graph_edges_pcx(MPI_COMM_WORLD, src_ranks);
  std::ranges::sort(dest_ranks0);
  std::ranges::sort(dest_ranks1);

  CHECK(dest_ranks0 == dest_ranks1);
}

void test_rank_split()
{
  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  constexpr int size_local = 100;
  const common::IndexMap idx_map
      = create_index_map(MPI_COMM_WORLD, size_local, (mpi_size - 1) * 3);

  {
    auto [dest_local, src_local] = idx_map.rank_type(MPI_COMM_TYPE_SHARED);
    REQUIRE(dest_local.size() <= idx_map.dest().size());
    REQUIRE(src_local.size() <= idx_map.src().size());
  }
}

void test_rank_weights()
{
  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  constexpr int size_local = 100;
  const common::IndexMap idx_map
      = create_index_map(MPI_COMM_WORLD, size_local, (mpi_size - 1) * 3);

  std::vector<std::int32_t> weights_src = idx_map.weights_src();
  std::vector<std::int32_t> weight_dest = idx_map.weights_dest();

  if (mpi_size > 1)
  {
    REQUIRE(weights_src == std::vector<std::int32_t>(1, (mpi_size - 1) * 3));
    REQUIRE(weight_dest == std::vector<std::int32_t>(1, (mpi_size - 1) * 3));
  }
  else
  {
    REQUIRE(weights_src.empty());
    REQUIRE(weight_dest.empty());
  }
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

TEST_CASE("Communication graph edges via consensus "
          "exchange",
          "[consensus_exchange]")
{
  CHECK_NOTHROW(test_consensus_exchange());
}

TEST_CASE("Split IndexMap communicator by type", "[index_map_comm_split]")
{
  CHECK_NOTHROW(test_rank_split());
}

TEST_CASE("IndexMap stats", "[index_map_stats]")
{
  CHECK_NOTHROW(test_rank_weights());
}
