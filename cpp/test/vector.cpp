// Copyright (C) 2021-2026 Chris Richardson and Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//
// Unit tests for Distributed la::Vector

#include <algorithm>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <complex>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/la/Vector.h>
#include <iterator>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

using namespace dolfinx;

namespace
{
// Stand-in for a device container: a distinct container template with
// the same shape as std::vector.
template <class T, class A = std::allocator<T>>
struct FakeDeviceVector : std::vector<T, A>
{
  using std::vector<T, A>::vector;
};

// clone_layout must rebind this vector's container to the new scalar
// type, not fall back to std::vector, or a device vector would clone to
// host storage while sharing a device scatterer.
void test_vector_clone_layout_container()
{
  using Host
      = la::Vector<double, std::vector<double>, std::vector<std::int32_t>>;
  static_assert(
      std::is_same_v<
          typename decltype(std::declval<Host>()
                                .clone_layout<std::int8_t>())::container_type,
          std::vector<std::int8_t>>);

  using Device
      = la::Vector<double, FakeDeviceVector<double>, std::vector<std::int32_t>>;
  static_assert(
      std::is_same_v<
          typename decltype(std::declval<Device>()
                                .clone_layout<std::int8_t>())::container_type,
          FakeDeviceVector<std::int8_t>>);

  // An explicit container is still honoured
  static_assert(
      std::is_same_v<
          typename decltype(std::declval<Device>()
                                .clone_layout<std::int8_t,
                                              std::vector<std::int8_t>>())::
              container_type,
          std::vector<std::int8_t>>);
}

template <typename T>
void test_vector()
{
  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  const int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  constexpr int size_local = 100;

  // Create some ghost entries on next process
  int num_ghosts = (mpi_size - 1) * 3;
  std::vector<std::int64_t> ghosts(num_ghosts);
  for (int i = 0; i < num_ghosts; ++i)
    ghosts[i] = (mpi_rank + 1) % mpi_size * size_local + i;

  const std::vector<int> global_ghost_owner(ghosts.size(),
                                            (mpi_rank + 1) % mpi_size);

  // Create an IndexMap
  auto index_map = std::make_shared<common::IndexMap>(
      MPI_COMM_WORLD, size_local, ghosts, global_ghost_owner);

  la::Vector<T> v(index_map, 1);
  std::ranges::fill(v.array(), 1.0);

  double norm2 = la::squared_norm(v);
  CHECK(norm2 == mpi_size * size_local);

  std::ranges::fill(v.array(), mpi_rank);

  double sumn2
      = size_local * (mpi_size - 1) * mpi_size * (2 * mpi_size - 1) / 6;
  CHECK(la::squared_norm(v) == sumn2);
  CHECK(la::norm(v, la::Norm::l2) == std::sqrt(sumn2));
  CHECK(la::inner_product(v, v) == sumn2);
  CHECK(la::norm(v, la::Norm::linf) == static_cast<T>(mpi_size - 1));
}

void test_vector_cast()
{
  using T = double;
  using U = float;

  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  const int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  constexpr int size_local = 100;

  // Create some ghost entries on next process
  int num_ghosts = (mpi_size - 1) * 3;
  std::vector<std::int64_t> ghosts(num_ghosts);
  for (int i = 0; i < num_ghosts; ++i)
    ghosts[i] = (mpi_rank + 1) % mpi_size * size_local + i;

  std::vector<int> global_ghost_owner(ghosts.size(), (mpi_rank + 1) % mpi_size);

  // Create an IndexMap
  auto index_map = std::make_shared<common::IndexMap>(
      MPI_COMM_WORLD, size_local, ghosts, global_ghost_owner);

  la::Vector<T> v(index_map, 1);
  std::ranges::fill(v.array(), 1);

  la::Vector<U, std::vector<U>, std::vector<std::int64_t>> v1(v);

  U norm2 = la::squared_norm(v1);
  CHECK(norm2 == mpi_size * size_local);

  std::ranges::fill(v1.array(), mpi_rank);

  U sumn2 = size_local * (mpi_size - 1) * mpi_size * (2 * mpi_size - 1) / 6;
  CHECK(la::squared_norm(v1) == sumn2);
  CHECK(la::norm(v1, la::Norm::l2) == std::sqrt(sumn2));
  CHECK(la::inner_product(v1, v1) == sumn2);
  CHECK(la::norm(v1, la::Norm::linf) == static_cast<U>(mpi_size - 1));
}
void test_vector_clone_layout()
{
  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  const int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  constexpr int size_local = 100;
  constexpr int bs = 3;

  // Create some ghost entries on next process
  int num_ghosts = (mpi_size - 1) * 3;
  std::vector<std::int64_t> ghosts(num_ghosts);
  for (int i = 0; i < num_ghosts; ++i)
    ghosts[i] = (mpi_rank + 1) % mpi_size * size_local + i;

  std::vector<int> global_ghost_owner(ghosts.size(), (mpi_rank + 1) % mpi_size);

  auto index_map = std::make_shared<common::IndexMap>(
      MPI_COMM_WORLD, size_local, ghosts, global_ghost_owner);

  la::Vector<double> v(index_map, bs);

  // Out of range for std::int8_t: the copy-converting constructor would
  // be undefined behaviour here
  std::ranges::fill(v.array(), 1e30);

  la::Vector<std::int8_t> marks = v.clone_layout<std::int8_t>();

  // Layout is shared, and the block size is inherited
  CHECK(marks.index_map() == v.index_map());
  CHECK(marks.bs() == bs);
  CHECK(marks.array().size() == v.array().size());

  // Entries are value-initialised, not copied
  CHECK(
      std::ranges::all_of(marks.array(), [](std::int8_t x) { return x == 0; }));

  // Storage is independent of the source
  std::ranges::fill(marks.array(), 1);
  CHECK(std::ranges::all_of(v.array(), [](double x) { return x == 1e30; }));

  // The shared scatterer serves the new value type
  std::ranges::fill(marks.array(), 0);
  std::fill_n(marks.array().begin(), bs * size_local,
              static_cast<std::int8_t>(mpi_rank + 1));
  marks.scatter_fwd();

  std::span<const std::int8_t> ghost_marks(
      std::next(marks.array().begin(), bs * size_local), marks.array().end());
  const std::int8_t expected
      = static_cast<std::int8_t>((mpi_rank + 1) % mpi_size + 1);
  CHECK(ghost_marks.size() == static_cast<std::size_t>(bs * num_ghosts));
  CHECK(std::ranges::all_of(ghost_marks, [expected](std::int8_t x)
                            { return x == expected; }));

  // Without an explicit type, a zeroed vector of the same type
  la::Vector<double> zeros = v.clone_layout();
  CHECK(zeros.index_map() == v.index_map());
  CHECK(zeros.bs() == bs);
  CHECK(std::ranges::all_of(zeros.array(), [](double x) { return x == 0; }));
}
} // namespace

TEMPLATE_TEST_CASE("Linear Algebra Vector", "[la_vector]", double,
                   std::complex<double>)
{
  CHECK_NOTHROW(test_vector<TestType>());
}

TEST_CASE("Linear Algebra Vector", "[la_vector]")
{
  CHECK_NOTHROW(test_vector_cast());
}

TEST_CASE("Linear Algebra Vector clone_layout", "[la_vector]")
{
  CHECK_NOTHROW(test_vector_clone_layout());
  CHECK_NOTHROW(test_vector_clone_layout_container());
}
