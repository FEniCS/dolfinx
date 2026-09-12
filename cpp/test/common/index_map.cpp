// Copyright (C) 2018-2026 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Scatterer.h>
#include <dolfinx/common/utils.h>
#include <functional>
#include <iostream>
#include <numeric>
#include <set>
#include <span>
#include <utility>
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

void test_scatter_fwd(int n, bool use_dtype)
{
  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  const int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  constexpr int size_local = 100;

  // Create an IndexMap
  const common::IndexMap idx_map
      = create_index_map(MPI_COMM_WORLD, size_local, (mpi_size - 1) * 3);
  std::int32_t num_ghosts = idx_map.num_ghosts();

  // Move, rather than copy, the Scatterer: a copy would duplicate the
  // communicators, which is collective
  common::Scatterer sct0(idx_map);
  common::Scatterer sct = std::move(sct0);

  // Create some data to scatter
  const std::int64_t val = 11;
  std::vector<std::int64_t> data_local(n * size_local, val * mpi_rank);
  std::vector<std::int64_t> data_ghost(n * num_ghosts, -1);

  // Scatter values to ghost and check value is correctly received
  {
    std::vector<std::int64_t> send_buffer(n * sct.local_indices_block().size());
    {
      auto& idx = sct.local_indices_block();
      for (std::size_t i = 0; i < idx.size(); ++i)
        for (int j = 0; j < n; ++j)
          send_buffer[i * n + j] = data_local[idx[i] * n + j];
    }
    std::vector<std::int64_t> recv_buffer(n
                                          * sct.remote_indices_block().size());
    MPI_Request request = MPI_REQUEST_NULL;
    if (use_dtype)
    {
      // Destroyed before scatter_fwd_end, since MPI keeps a datatype
      // alive until communication using it has completed
      const dolfinx::MPI::Datatype<std::int64_t> type(n);
      sct.scatter_fwd_begin_dtype(send_buffer.data(), recv_buffer.data(),
                                  type.type(), request);
    }
    else
      sct.scatter_fwd_begin(send_buffer.data(), recv_buffer.data(), n, request);
    sct.scatter_fwd_end(request);
    {
      auto& idx = sct.remote_indices_block();
      for (std::size_t i = 0; i < idx.size(); ++i)
        for (int j = 0; j < n; ++j)
          data_ghost[idx[i] * n + j] = recv_buffer[i * n + j];
    }
    CHECK((int)data_ghost.size() == n * num_ghosts);
    CHECK(std::ranges::all_of(
        data_ghost, [&val, &mpi_rank, &mpi_size](auto i)
        { return i == val * ((mpi_rank + 1) % mpi_size); }));
  }
}

void test_scatter_rev(bool use_dtype)
{
  // Block size
  auto n = GENERATE(1, 5, 10);

  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  constexpr int size_local = 100;

  // Create an IndexMap
  const common::IndexMap idx_map
      = create_index_map(MPI_COMM_WORLD, size_local, (mpi_size - 1) * 3);
  std::int32_t num_ghosts = idx_map.num_ghosts();

  common::Scatterer<std::vector<std::int32_t>> sct(idx_map);
  {
    common::Scatterer<std::vector<std::int64_t>> sct2(sct);
  }

  // Start a reverse scatter through either the block size or the MPI
  // datatype interface
  auto rev_begin_fn
      = [&sct, n, use_dtype](const std::int64_t* send_buffer,
                             std::int64_t* recv_buffer, MPI_Request& request)
  {
    if (use_dtype)
    {
      const dolfinx::MPI::Datatype<std::int64_t> type(n);
      sct.scatter_rev_begin_dtype(send_buffer, recv_buffer, type.type(),
                                  request);
    }
    else
      sct.scatter_rev_begin(send_buffer, recv_buffer, n, request);
  };

  auto pack_fn = [n](auto&& in, auto&& idx, auto&& out)
  {
    for (std::size_t i = 0; i < idx.size(); ++i)
      for (int j = 0; j < n; ++j)
        out[i * n + j] = in[idx[i] * n + j];
  };
  auto unpack_fn = [n](auto&& in, auto&& idx, auto&& out, auto op)
  {
    for (std::size_t i = 0; i < idx.size(); ++i)
      for (int j = 0; j < n; ++j)
      {
        auto& o = out[idx[i] * n + j];
        o = op(o, in[i * n + j]);
      }
  };

  // Create some data, setting ghost values
  std::int64_t value = 15;
  std::vector<std::int64_t> data_local(n * size_local, 0);
  std::vector<std::int64_t> data_ghost(n * num_ghosts, value);
  {
    MPI_Request request = MPI_REQUEST_NULL;
    std::vector<std::int64_t> send_buffer(n * sct.remote_indices_block().size(),
                                          0);
    pack_fn(data_ghost, sct.remote_indices_block(), send_buffer);
    std::vector<std::int64_t> recv_buffer(n * sct.local_indices_block().size(),
                                          0);
    rev_begin_fn(send_buffer.data(), recv_buffer.data(), request);
    sct.scatter_rev_end(request);
    unpack_fn(recv_buffer, sct.local_indices_block(), data_local,
              std::plus<>{});

    std::int64_t sum;
    CHECK((int)data_local.size() == n * size_local);
    sum = std::reduce(data_local.begin(), data_local.end(), 0);
    CHECK(sum == n * value * num_ghosts);
  }

  // Repeat, to check accumulation onto the already-populated
  // data_local rather than overwriting it
  {
    MPI_Request request = MPI_REQUEST_NULL;
    std::vector<std::int64_t> send_buffer(n * sct.remote_indices_block().size(),
                                          0);
    pack_fn(data_ghost, sct.remote_indices_block(), send_buffer);
    std::vector<std::int64_t> recv_buffer(n * sct.local_indices_block().size(),
                                          0);
    rev_begin_fn(send_buffer.data(), recv_buffer.data(), request);
    sct.scatter_rev_end(request);
    unpack_fn(recv_buffer, sct.local_indices_block(), data_local,
              std::plus<>{});

    std::int64_t sum = std::reduce(data_local.begin(), data_local.end(), 0);
    CHECK(sum == 2 * n * value * num_ghosts);
  }
}

void test_scatter_with_isolated_rank()
{
  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  const int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  if (mpi_size < 3)
    return;

  std::array<std::vector<int>, 2> src_dest;
  std::vector<std::int64_t> ghosts;
  std::vector<int> owners;
  if (mpi_rank == 0)
    src_dest[1] = {1};
  else if (mpi_rank == 1)
  {
    src_dest[0] = {0};
    ghosts = {0};
    owners = {0};
  }

  const common::IndexMap map(MPI_COMM_WORLD, 1, src_dest, ghosts, owners);
  const common::Scatterer scatterer(map);

  {
    std::vector<std::int64_t> send_buffer(
        scatterer.local_indices_block().size(), 17);
    std::vector<std::int64_t> recv_buffer(
        scatterer.remote_indices_block().size());
    MPI_Request request = MPI_REQUEST_NULL;
    scatterer.scatter_fwd_begin(send_buffer.data(), recv_buffer.data(), 1,
                                request);
    CHECK(request != MPI_REQUEST_NULL);
    scatterer.scatter_fwd_end(request);
    if (mpi_rank == 1)
      CHECK(recv_buffer == std::vector<std::int64_t>{17});
  }

  {
    std::vector<std::int64_t> send_buffer(
        scatterer.remote_indices_block().size(), 29);
    std::vector<std::int64_t> recv_buffer(
        scatterer.local_indices_block().size());
    MPI_Request request = MPI_REQUEST_NULL;
    scatterer.scatter_rev_begin(send_buffer.data(), recv_buffer.data(), 1,
                                request);
    CHECK(request != MPI_REQUEST_NULL);
    scatterer.scatter_rev_end(request);
    if (mpi_rank == 0)
      CHECK(recv_buffer == std::vector<std::int64_t>{29});
  }
}

// A copy of a Scatterer duplicates the communicators, so the copy must
// describe the same communication pattern as the original. A move takes
// the communicators over.
void test_scatter_copy_move()
{
  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  const int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  // Must be at least the ghost count, so that every ghost index is in
  // the owning rank's range
  constexpr int size_local = 100;
  const common::IndexMap idx_map
      = create_index_map(MPI_COMM_WORLD, size_local, (mpi_size - 1) * 3);

  const std::int64_t val = 11;
  const std::vector<std::int64_t> data_local(size_local, val * mpi_rank);

  // Forward scatter data_local through `s`, returning the ghost values
  // received
  auto fwd = [&data_local](auto&& s)
  {
    const auto& idx_local = s.local_indices_block();
    std::vector<std::int64_t> send_buffer(idx_local.size());
    for (std::size_t i = 0; i < idx_local.size(); ++i)
      send_buffer[i] = data_local[idx_local[i]];

    std::vector<std::int64_t> recv_buffer(s.remote_indices_block().size(), -1);
    MPI_Request request = MPI_REQUEST_NULL;
    s.scatter_fwd_begin(send_buffer.data(), recv_buffer.data(), 1, request);
    s.scatter_fwd_end(request);
    return recv_buffer;
  };

  common::Scatterer sct(idx_map);
  const std::vector<std::int64_t> expected = fwd(sct);
  // `val` is a constant expression, so it needs no capture
  CHECK(
      std::ranges::all_of(expected, [mpi_rank, mpi_size](std::int64_t i)
                          { return i == val * ((mpi_rank + 1) % mpi_size); }));

  // Copy, and copy to a different index container type
  {
    const common::Scatterer sct_copy(sct);
    CHECK(sct_copy.local_indices_block() == sct.local_indices_block());
    CHECK(sct_copy.remote_indices_block() == sct.remote_indices_block());
    CHECK(fwd(sct_copy) == expected);

    const common::Scatterer<std::vector<std::int64_t>> sct_cast(sct);
    CHECK(fwd(sct_cast) == expected);
  }

  // Move construction, then move assignment onto a Scatterer that
  // already holds communicators
  {
    common::Scatterer sct_move(std::move(sct));
    CHECK(fwd(sct_move) == expected);

    common::Scatterer sct_target(idx_map);
    sct_target = std::move(sct_move);
    CHECK(fwd(sct_target) == expected);
  }
}

// On a single-rank communicator there are no neighbours, so a scatter
// performs no communication and leaves the request as MPI_REQUEST_NULL
void test_scatter_single_rank()
{
  const common::IndexMap map(MPI_COMM_SELF, 4);
  const common::Scatterer sct(map);
  CHECK(sct.local_indices_block().empty());
  CHECK(sct.remote_indices_block().empty());

  std::vector<std::int64_t> send_buffer(4, 0), recv_buffer(4, 0);
  MPI_Request request = MPI_REQUEST_NULL;
  sct.scatter_fwd_begin(send_buffer.data(), recv_buffer.data(), 1, request);
  CHECK(request == MPI_REQUEST_NULL);
  sct.scatter_fwd_end(request);

  sct.scatter_rev_begin(send_buffer.data(), recv_buffer.data(), 1, request);
  CHECK(request == MPI_REQUEST_NULL);
  sct.scatter_rev_end(request);
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

/// Check the one-shot common::scatter_fwd/scatter_rev helpers. Only
/// std::int8_t is exercised here: the other host types are covered
/// through the Python bindings, which delegate to these helpers, but are
/// instantiated only for int64/float/double.
void test_scatter_helpers_int8()
{
  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  const int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  constexpr int size_local = 100;
  constexpr int bs = 2;

  const common::IndexMap idx_map
      = create_index_map(MPI_COMM_WORLD, size_local, (mpi_size - 1) * 3);
  const std::int32_t num_ghosts = idx_map.num_ghosts();
  const common::Scatterer sct(idx_map);

  // Forward scatter the owning rank (+1, to keep zero distinguishable
  // from an untouched buffer)
  const std::vector<std::int8_t> data_local(
      bs * size_local, static_cast<std::int8_t>(1 + mpi_rank % 5));
  std::vector<std::int8_t> data_ghost(bs * num_ghosts, 0);
  common::scatter_fwd<std::int8_t>(sct,
                                   std::span<const std::int8_t>(data_local),
                                   std::span<std::int8_t>(data_ghost), bs);
  const std::int8_t expected
      = static_cast<std::int8_t>(1 + (mpi_rank + 1) % mpi_size % 5);
  CHECK(std::ranges::all_of(data_ghost, [expected](std::int8_t v)
                            { return v == expected; }));

  // Reverse scatter, accumulating each ghost value onto its owner. Every
  // ghost index is ghosted by exactly one rank, so each contribution
  // arrives once.
  std::ranges::fill(data_ghost, std::int8_t(2));
  std::vector<std::int8_t> data_owned(bs * size_local, 0);
  common::scatter_rev<std::int8_t>(sct, std::span<std::int8_t>(data_owned),
                                   std::span<const std::int8_t>(data_ghost), bs,
                                   std::plus<std::int8_t>());
  CHECK(std::accumulate(data_owned.begin(), data_owned.end(), std::int64_t(0))
        == 2 * bs * num_ghosts);
}
} // namespace

TEST_CASE("Scatter forward using IndexMap", "[index_map_scatter_fwd]")
{
  auto n = GENERATE(1, 5, 10);
  auto use_dtype = GENERATE(false, true);
  CHECK_NOTHROW(test_scatter_fwd(n, use_dtype));
}

TEST_CASE("Scatter reverse using IndexMap", "[index_map_scatter_rev]")
{
  auto use_dtype = GENERATE(false, true);
  CHECK_NOTHROW(test_scatter_rev(use_dtype));
}

TEST_CASE("Scatter with a copied and a moved Scatterer", "[index_map_scatter]")
{
  CHECK_NOTHROW(test_scatter_copy_move());
}

TEST_CASE("Scatter on a single rank", "[index_map_scatter]")
{
  CHECK_NOTHROW(test_scatter_single_rank());
}

TEST_CASE("Scatter with an isolated rank", "[index_map_scatter]")
{
  CHECK_NOTHROW(test_scatter_with_isolated_rank());
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

TEST_CASE("One-shot scatter helpers", "[index_map_scatter_helpers]")
{
  CHECK_NOTHROW(test_scatter_helpers_int8());
}
