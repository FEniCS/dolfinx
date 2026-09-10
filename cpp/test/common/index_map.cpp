// Copyright (C) 2018-2026 Chris Richardson, Garth N. Wells and Jack S. Hale
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
#include <dolfinx/common/ScatterPattern.h>
#include <dolfinx/common/Scatterer.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/la/Vector.h>
#include <iostream>
#include <iterator>
#include <memory>
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
    sct.scatter_fwd_end(request);
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

  // A scatterer built directly on a wider index container expands the
  // pattern indices itself, rather than converting an expanded copy
  {
    common::Scatterer<std::vector<std::int64_t>> sct2(idx_map, n);
    CHECK(std::ranges::equal(sct2.local_indices(), sct.local_indices()));
    CHECK(std::ranges::equal(sct2.remote_indices(), sct.remote_indices()));
  }

  // Preconditions on the block size and the pattern
  CHECK_THROWS_AS(common::Scatterer<>(idx_map.scatter_pattern(), 0),
                  std::invalid_argument);
  CHECK_THROWS_AS(common::Scatterer<>(idx_map.scatter_pattern(), -1),
                  std::invalid_argument);
  CHECK_THROWS_AS(common::Scatterer<>(nullptr, 1), std::invalid_argument);

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
    std::vector<std::int64_t> send_buffer(sct.remote_indices().size(), 0);
    pack_fn(data_ghost, sct.remote_indices(), send_buffer);
    std::vector<std::int64_t> recv_buffer(sct.local_indices().size(), 0);
    sct.scatter_rev_begin(send_buffer.data(), recv_buffer.data(), request);
    sct.scatter_rev_end(request);
    unpack_fn(recv_buffer, sct.local_indices(), data_local, std::plus<>{});

    std::int64_t sum;
    CHECK((int)data_local.size() == n * size_local);
    sum = std::reduce(data_local.begin(), data_local.end(), 0);
    CHECK(sum == n * value * num_ghosts);
  }

  // Repeat, to check accumulation onto the already-populated
  // data_local rather than overwriting it
  {
    MPI_Request request = MPI_REQUEST_NULL;
    std::vector<std::int64_t> send_buffer(sct.remote_indices().size(), 0);
    pack_fn(data_ghost, sct.remote_indices(), send_buffer);
    std::vector<std::int64_t> recv_buffer(sct.local_indices().size(), 0);
    sct.scatter_rev_begin(send_buffer.data(), recv_buffer.data(), request);
    sct.scatter_rev_end(request);
    unpack_fn(recv_buffer, sct.local_indices(), data_local, std::plus<>{});

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
  const common::Scatterer scatterer(map, 1);

  {
    std::vector<std::int64_t> send_buffer(scatterer.local_indices().size(), 17);
    std::vector<std::int64_t> recv_buffer(scatterer.remote_indices().size());
    MPI_Request request = MPI_REQUEST_NULL;
    scatterer.scatter_fwd_begin(send_buffer.data(), recv_buffer.data(),
                                request);
    CHECK(request != MPI_REQUEST_NULL);
    scatterer.scatter_fwd_end(request);
    if (mpi_rank == 1)
      CHECK(recv_buffer == std::vector<std::int64_t>{17});
  }

  {
    std::vector<std::int64_t> send_buffer(scatterer.remote_indices().size(),
                                          29);
    std::vector<std::int64_t> recv_buffer(scatterer.local_indices().size());
    MPI_Request request = MPI_REQUEST_NULL;
    scatterer.scatter_rev_begin(send_buffer.data(), recv_buffer.data(),
                                request);
    CHECK(request != MPI_REQUEST_NULL);
    scatterer.scatter_rev_end(request);
    if (mpi_rank == 0)
      CHECK(recv_buffer == std::vector<std::int64_t>{29});
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
void test_scatter_pattern_shared()
{
  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  constexpr int size_local = 100;
  auto map = std::make_shared<const common::IndexMap>(
      create_index_map(MPI_COMM_WORLD, size_local, (mpi_size - 1) * 3));

  // The pattern is built once and returned to every subsequent caller
  std::shared_ptr<const common::ScatterPattern> pattern
      = map->scatter_pattern();
  CHECK(pattern.get() == map->scatter_pattern().get());
  CHECK(pattern.use_count() == 2);

  // Scatterers share the pattern, whatever their block size or index
  // container type. Two neighbourhood communicators are created for the
  // index map, not two per Scatterer.
  {
    common::Scatterer<std::vector<std::int32_t>> sct0(*map, 1);
    common::Scatterer<std::vector<std::int32_t>> sct1(*map, 3);
    CHECK(pattern.use_count() == 4);

    // The cast-copy constructor shares the pattern rather than
    // duplicating the communicators
    common::Scatterer<std::vector<std::int64_t>> sct2(sct0);
    CHECK(pattern.use_count() == 5);
  }
  CHECK(pattern.use_count() == 2);

  // A code that creates many vectors over one index map creates no
  // extra communicators. This is the regression test for
  // https://github.com/FEniCS/dolfinx/issues/3065.
  {
    constexpr int num_vectors = 16;
    std::vector<la::Vector<double>> vectors;
    vectors.reserve(num_vectors);
    for (int i = 0; i < num_vectors; ++i)
      vectors.emplace_back(map, 2);
    CHECK(pattern.use_count() == 2 + num_vectors);

    // Cloning a layout shares the scatterer, so does not add a
    // reference to the pattern
    la::Vector<std::int8_t> marks = vectors.front().clone_layout<std::int8_t>();
    CHECK(pattern.use_count() == 2 + num_vectors);
  }
  CHECK(pattern.use_count() == 2);
}
void test_scatter_overlap()
{
  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  const int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  constexpr int size_local = 100;
  auto map = std::make_shared<const common::IndexMap>(
      create_index_map(MPI_COMM_WORLD, size_local, (mpi_size - 1) * 3));
  std::int32_t num_ghosts = map->num_ghosts();

  // Vectors over one index map, and so over one pair of neighbourhood
  // communicators. Each holds its own buffers and MPI_Request.
  constexpr int num_vectors = 4;
  std::vector<la::Vector<double>> v;
  v.reserve(num_vectors);
  for (int k = 0; k < num_vectors; ++k)
  {
    v.emplace_back(map, 1);
    std::ranges::fill_n(v.back().array().begin(), size_local,
                        100.0 * mpi_rank + k);
  }

  // Every forward scatter in flight at once on the shared communicator.
  // Nonblocking collectives on one communicator are matched in issue
  // order, which is identical on every rank here.
  for (auto& x : v)
    x.scatter_fwd_begin();
  for (auto& x : v)
    x.scatter_fwd_end();

  // Each vector must receive its own data, not another's
  const double owner = 100.0 * ((mpi_rank + 1) % mpi_size);
  for (int k = 0; k < num_vectors; ++k)
  {
    const std::vector<double>& x = v[k].array();
    for (std::int32_t i = 0; i < num_ghosts; ++i)
      CHECK(x[size_local + i] == owner + k);
  }

  // A forward and a reverse scatter overlap without an ordering
  // constraint between them: they use different communicators.
  std::ranges::fill(v[0].array(), 0.0);
  std::ranges::fill_n(v[0].array().begin(), size_local, 7.0);
  std::ranges::fill(v[1].array(), 0.0);
  std::ranges::fill_n(std::next(v[1].array().begin(), size_local), num_ghosts,
                      1.0);
  // out[idx[i]] = out[idx[i]] + in[i]
  auto unpack_add = [](std::vector<std::int32_t>::const_iterator idx_first,
                       std::vector<std::int32_t>::const_iterator idx_last,
                       const auto in_first, auto out_first)
  {
    for (auto idx = idx_first; idx != idx_last; ++idx)
    {
      std::size_t d = std::ranges::distance(idx_first, idx);
      auto& out = *std::next(out_first, *idx);
      out = out + *std::next(in_first, d);
    }
  };

  v[0].scatter_fwd_begin();
  v[1].scatter_rev_begin();
  v[0].scatter_fwd_end();
  v[1].scatter_rev_end(unpack_add);
  for (std::int32_t i = 0; i < num_ghosts; ++i)
    CHECK(v[0].array()[size_local + i] == 7.0);
  if (mpi_size > 1)
  {
    // Rank r owns the indices ghosted by rank r - 1, three per rank
    const std::vector<double>& x = v[1].array();
    double received = 0;
    for (std::int32_t i = 0; i < size_local; ++i)
      received += x[i];
    CHECK(received == static_cast<double>(num_ghosts));
  }
}
void test_private_scatter_pattern()
{
  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  const int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  constexpr int size_local = 100;
  auto map = std::make_shared<const common::IndexMap>(
      create_index_map(MPI_COMM_WORLD, size_local, (mpi_size - 1) * 3));
  std::int32_t num_ghosts = map->num_ghosts();

  std::shared_ptr<const common::ScatterPattern> shared = map->scatter_pattern();

  // A caller that wants its scatters unordered with respect to every
  // other scatter on this map can build a private pattern, and with it
  // a private pair of communicators.
  auto priv = std::make_shared<const common::ScatterPattern>(*map);
  CHECK(priv.get() != shared.get());
  if (mpi_size > 1)
    CHECK(priv->comm0() != shared->comm0());

  auto sct = std::make_shared<const common::Scatterer<>>(priv, 1);
  la::Vector<double> v(map, 1, sct);

  // The vector did not take the map's shared pattern
  CHECK(shared.use_count() == 2);

  // ... and scatters correctly on its own communicators
  std::ranges::fill_n(v.array().begin(), size_local, 1.0 * mpi_rank);
  v.scatter_fwd();
  const double owner = static_cast<double>((mpi_rank + 1) % mpi_size);
  for (std::int32_t i = 0; i < num_ghosts; ++i)
    CHECK(v.array()[size_local + i] == owner);
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

TEST_CASE("IndexMap scatter pattern is shared", "[index_map_scatter_pattern]")
{
  CHECK_NOTHROW(test_scatter_pattern_shared());
}

TEST_CASE("Overlapping scatters share a communicator",
          "[index_map_scatter_overlap]")
{
  CHECK_NOTHROW(test_scatter_overlap());
}

TEST_CASE("Opt out of the shared scatter pattern",
          "[index_map_private_pattern]")
{
  CHECK_NOTHROW(test_private_scatter_pattern());
}
