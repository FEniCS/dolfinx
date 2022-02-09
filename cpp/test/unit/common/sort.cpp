// Copyright (C) 2021 Igor A. Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch2/catch.hpp>
#include <dolfinx/common/sort.h>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

TEMPLATE_TEST_CASE("Test radix sort", "[vector][template]", std::int32_t,
                   std::int64_t)
{
  auto vec_size = GENERATE(100, 1000, 10000);
  std::vector<TestType> vec;
  vec.reserve(vec_size);

  // Gererate a vector of ints with a Uniform Int distribution
  std::uniform_int_distribution<TestType> distribution(0, 10000);
  std::mt19937 engine;
  auto generator = std::bind(distribution, engine);
  std::generate_n(std::back_inserter(vec), vec_size, generator);

  // Sort vector using radix sort
  dolfinx::radix_sort(xtl::span(vec));

  // Check if vector is sorted
  REQUIRE(std::is_sorted(vec.begin(), vec.end()));
}

TEST_CASE("Test argsort bitset")
{
  auto size = GENERATE(100, 1000, 10000);

  xt::xtensor<std::int32_t, 2> arr = xt::empty<std::int32_t>({size, 2});

  // Gererate a vector of ints with a Uniform Int distribution
  std::uniform_int_distribution<std::int32_t> distribution(0, 10000);
  std::mt19937 engine;
  auto generator = std::bind(distribution, engine);
  std::generate(arr.begin(), arr.end(), generator);

  std::vector<std::int32_t> perm = dolfinx::sort_by_perm(arr);
  REQUIRE(perm.size() == arr.shape(0));

  // Sort by perm using to std::lexicographical_compare
  std::vector<int> index(arr.shape(0));
  std::iota(index.begin(), index.end(), 0);
  std::sort(index.begin(), index.end(),
            [&arr](int a, int b)
            {
              return std::lexicographical_compare(
                  xt::row(arr, a).begin(), xt::row(arr, a).end(),
                  xt::row(arr, b).begin(), xt::row(arr, b).end());
            });

  // Requiring equality of permutation vectors is not a good test, because
  // std::sort is not stable, so we compare the effect on the actual array.
  for (std::size_t i = 0; i < perm.size(); i++)
    REQUIRE((xt::row(arr, perm[i]) == xt::row(arr, index[i])));
}
