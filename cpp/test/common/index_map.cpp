// Copyright (C) 2018 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Scatterer.h>
#include <dolfinx/common/utils.h>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
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
        data_ghost, [&val, &mpi_rank, &mpi_size](auto i)
        { return i == val * ((mpi_rank + 1) % mpi_size); }));
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

  common::Scatterer<std::vector<std::int32_t>> sct(idx_map, n);
  {
    common::Scatterer<std::vector<std::int64_t>> sct2(sct);
  }

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
    MPI_Request request = MPI_REQUEST_NULL;
    std::vector<std::int64_t> remote_buffer(sct.remote_indices().size(), 0);
    std::vector<std::int64_t> send_buffer(sct.local_indices().size(), 0);
    pack_fn(data_ghost, sct.remote_indices(), send_buffer);
    std::vector<std::int64_t> recv_buffer(sct.remote_indices().size(), 0);
    sct.scatter_rev_begin(send_buffer.data(), recv_buffer.data(), request);
    sct.scatter_end(request);
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

void test_index_map_preconditions()
{
#ifndef NDEBUG
  CHECK_THROWS_AS(common::IndexMap(MPI_COMM_WORLD, -1), std::invalid_argument);

  // A ghosts/owners length mismatch would otherwise be an out-of-bounds
  // access in internal communication setup.
  const std::vector<std::int64_t> mismatched_ghosts = {0, 1};
  const std::vector<int> mismatched_owners = {0};
  CHECK_THROWS_AS(
      common::IndexMap(MPI_COMM_WORLD, 1, mismatched_ghosts, mismatched_owners),
      std::invalid_argument);
  const std::array<std::vector<int>, 2> empty_src_dest = {};
  CHECK_THROWS_AS(common::IndexMap(MPI_COMM_WORLD, 1, empty_src_dest,
                                   mismatched_ghosts, mismatched_owners),
                  std::invalid_argument);

  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  const int owner = (dolfinx::MPI::rank(MPI_COMM_WORLD) + 1) % mpi_size;
  const std::vector<std::int64_t> ghosts = {0};
  const std::vector<int> owners = {owner};
  const std::array<std::vector<int>, 2> src_dest = {};
  CHECK_THROWS_AS(common::IndexMap(MPI_COMM_WORLD, 1, src_dest, ghosts, owners),
                  std::invalid_argument);

  if (mpi_size > 1)
  {
    const int rank = dolfinx::MPI::rank(MPI_COMM_WORLD);

    // A precondition failure on one rank is reported on all ranks before
    // either constructor enters its communication path.
    CHECK_THROWS_AS(common::IndexMap(MPI_COMM_WORLD, rank == 0 ? -1 : 1),
                    std::invalid_argument);
    const std::vector<std::int64_t> uneven_ghosts
        = rank == 0 ? std::vector<std::int64_t>{0}
                    : std::vector<std::int64_t>{};
    const std::vector<int> no_owners;
    CHECK_THROWS_AS(
        common::IndexMap(MPI_COMM_WORLD, 1, uneven_ghosts, no_owners),
        std::invalid_argument);

    const std::vector<std::int64_t> ghost_owned_locally = {rank};
    const std::vector<int> remote_owner = {(rank + 1) % mpi_size};
    const std::vector<int> destination = {(rank + mpi_size - 1) % mpi_size};
    const std::array<std::vector<int>, 2> invalid_src_dest
        = {remote_owner, destination};
    CHECK_THROWS_AS(
        common::IndexMap(MPI_COMM_WORLD, 1, ghost_owned_locally, remote_owner),
        std::invalid_argument);
    CHECK_THROWS_AS(common::IndexMap(MPI_COMM_WORLD, 1, invalid_src_dest,
                                     ghost_owned_locally, remote_owner),
                    std::invalid_argument);

    const std::vector<std::int64_t> out_of_range_ghost = {mpi_size};
    CHECK_THROWS_AS(common::IndexMap(MPI_COMM_WORLD, 1, invalid_src_dest,
                                     out_of_range_ghost, remote_owner),
                    std::invalid_argument);

    if (mpi_size > 2)
    {
      const std::vector<std::int64_t> ghost_owned_remotely = {remote_owner[0]};
      const std::array<std::vector<int>, 2> mismatched_src_dest
          = {remote_owner, remote_owner};
      CHECK_THROWS_AS(common::IndexMap(MPI_COMM_WORLD, 1, mismatched_src_dest,
                                       ghost_owned_remotely, remote_owner),
                      std::invalid_argument);
    }
  }
#endif

  const common::IndexMap map(MPI_COMM_WORLD, 1);
#ifndef NDEBUG
  const std::vector<std::int32_t> duplicate_indices = {0, 0};
  const std::vector<std::int32_t> out_of_range_indices = {1};
  CHECK_THROWS_AS(common::create_sub_index_map(map, duplicate_indices),
                  std::invalid_argument);
  CHECK_THROWS_AS(common::compute_owned_indices(duplicate_indices, map),
                  std::invalid_argument);
  CHECK_THROWS_AS(common::create_sub_index_map(map, out_of_range_indices),
                  std::invalid_argument);
  CHECK_THROWS_AS(common::compute_owned_indices(out_of_range_indices, map),
                  std::invalid_argument);
#endif

  const std::vector<std::int32_t> valid_indices = {0};
  auto [submap, submap_to_map]
      = common::create_sub_index_map(map, valid_indices);
  CHECK(submap.size_local() == 1);
  CHECK(submap_to_map == valid_indices);
}

void test_compute_owned_indices()
{
  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  if (mpi_size == 1)
  {
    const common::IndexMap map(MPI_COMM_WORLD, 1);
    const std::vector<std::int32_t> selected;
    CHECK(common::compute_owned_indices(selected, map).empty());
    return;
  }

  const int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  const int owner = (mpi_rank + 1) % mpi_size;
  const std::vector<std::int64_t> ghosts = {owner};
  const std::vector<int> owners = {owner};
  const common::IndexMap map(MPI_COMM_WORLD, 1, ghosts, owners);

  // Each rank selects its only ghost. Its predecessor therefore selects the
  // local entry owned by this rank.
  const std::vector<std::int32_t> selected = {1};
  const std::vector<std::int32_t> expected = {0};
  CHECK(common::compute_owned_indices(selected, map) == expected);
}

void test_local_global_index_conversion()
{
  const common::IndexMap map(MPI_COMM_WORLD, 2);
  std::vector<std::int64_t> global(1);
  const std::vector<std::int32_t> local_two = {0, 1};
  const std::vector<std::int32_t> local_negative = {-1};
  const std::vector<std::int32_t> local_out_of_range = {2};
  CHECK_THROWS_AS(map.local_to_global(local_two, global),
                  std::invalid_argument);
  CHECK_THROWS_AS(map.local_to_global(local_negative, global),
                  std::out_of_range);
  CHECK_THROWS_AS(map.local_to_global(local_out_of_range, global),
                  std::out_of_range);

  std::vector<std::int64_t> global_larger(3, -1);
  CHECK_NOTHROW(map.local_to_global(local_two, global_larger));
  CHECK(global_larger[0] == map.local_range()[0]);
  CHECK(global_larger[1] == map.local_range()[0] + 1);
  CHECK(global_larger[2] == -1);

  std::vector<std::int32_t> local(1);
  const std::vector<std::int64_t> global_two = {0, 1};
  CHECK_THROWS_AS(map.global_to_local(global_two, local),
                  std::invalid_argument);
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

TEST_CASE("IndexMap preconditions", "[index_map_preconditions]")
{
  CHECK_NOTHROW(test_index_map_preconditions());
}

TEST_CASE("Compute owned IndexMap indices", "[index_map_owned_indices]")
{
  CHECK_NOTHROW(test_compute_owned_indices());
}

TEST_CASE("IndexMap local/global conversions", "[index_map_conversions]")
{
  CHECK_NOTHROW(test_local_global_index_conversion());
}
