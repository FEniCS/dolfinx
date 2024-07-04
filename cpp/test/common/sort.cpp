// Copyright (C) 2021 Igor A. Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <dolfinx/common/sort.h>
#include <functional>
#include <random>

TEMPLATE_TEST_CASE("Test radix sort", "[vector][template]", std::int32_t,
                   std::int64_t)
{
  auto vec_size = GENERATE(100, 1000, 10000);
  std::vector<TestType> vec;
  vec.reserve(vec_size);

  // Generate a vector of ints with a Uniform Int distribution
  std::uniform_int_distribution<TestType> distribution(0, 10000);
  std::mt19937 engine;
  auto generator = std::bind(distribution, engine);
  std::generate_n(std::back_inserter(vec), vec_size, generator);

  // Sort vector using radix sort
  dolfinx::radix_sort(std::span(vec));

  // Check if vector is sorted
  REQUIRE(std::ranges::is_sorted(vec));
}

TEST_CASE("Test argsort bitset")
{
  auto shape0 = GENERATE(100, 1000, 10000);
  constexpr int shape1 = 2;

  std::vector<std::int32_t> arr(shape0 * shape1);

  // Generate a vector of ints with a Uniform Int distribution
  std::uniform_int_distribution<std::int32_t> distribution(0, 10000);
  std::mt19937 engine;
  auto generator = std::bind(distribution, engine);
  std::ranges::generate(arr, generator);

  std::vector<std::int32_t> perm
      = dolfinx::sort_by_perm<std::int32_t>(arr, shape1);
  REQUIRE((int)perm.size() == shape0);

  // Sort by perm using to std::lexicographical_compare
  std::vector<int> index(shape0);
  std::iota(index.begin(), index.end(), 0);
  std::ranges::sort(index,
                    [&arr](int a, int b)
                    {
                      auto it0 = std::next(arr.begin(), shape1 * a);
                      auto it1 = std::next(arr.begin(), shape1 * b);
                      return std::lexicographical_compare(
                          it0, std::next(it0, shape1), it1,
                          std::next(it1, shape1));
                    });

  // Requiring equality of permutation vectors is not a good test, because
  // std::sort is not stable, so we compare the effect on the actual array.
  for (std::size_t i = 0; i < perm.size(); i++)
  {
    REQUIRE(std::equal(arr.data() + shape1 * perm[i],
                       arr.data() + shape1 * perm[i] + shape1,
                       arr.data() + shape1 * index[i]));
  }
}
