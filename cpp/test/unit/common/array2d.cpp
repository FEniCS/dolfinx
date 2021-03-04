// Copyright (C) 2021 Igor Baratta
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch.hpp>
#include <dolfinx/common/ndarray.h>
#include <numeric>
#include <vector>

using namespace dolfinx;

namespace
{
void test_array2d()
{
  // Define array shape
  std::size_t ni = GENERATE(3, 100, 300);
  std::size_t nj = GENERATE(3, 100, 300);

  std::array<std::size_t, 2> shape = {ni, nj};
  ndarray<std::size_t, 2> arr(shape);

  CHECK(arr.shape[0] == ni);
  CHECK(arr.shape[1] == nj);

  for (int i = 0; i < ni; i++)
  {
    auto row = arr.row(i);
    std::iota(row.begin(), row.end(), 0);
  }

  for (std::size_t i = 0; i < arr.shape[0]; i++)
    for (std::size_t j = 0; j < arr.shape[1]; j++)
      REQUIRE(arr(i, j) == j);
}

void test_dot_product()
{
  std::array<std::size_t, 2> shape1 = {3, 100};
  std::array<std::size_t, 2> shape2 = {100, 3};

  ndarray<double, 2> arr1(shape1, 1.);
  ndarray<double, 2> arr2(shape2, 2.);

  // transpose arr2 and use vector data structure
  std::vector<double> data_transpose(300);
  ndspan<double, 2> arr2_T(data_transpose.data(), shape1);
  for (std::size_t i = 0; i < arr2.shape[0]; i++)
    for (std::size_t j = 0; j < arr2.shape[1]; j++)
      arr2_T(j, i) = arr2(i, j);

  for (auto e : data_transpose)
    CHECK(e == 2.0);

  std::array<double, 9> result = {0};
  ndspan<double, 2> view(result.data(), {3, 3});

  for (std::size_t i = 0; i < view.shape[0]; i++)
    for (std::size_t j = 0; j < view.shape[1]; j++)
      view(i, j) = std::inner_product(arr1.row(i).begin(), arr1.row(i).end(),
                                      arr2_T.row(j).begin(), 0.0);

  for (auto e : result)
    CHECK(e == 200.0);
}

TEST_CASE("Test row acces and operator overloading.", "[test_array2d]")
{
  CHECK_NOTHROW(test_array2d());
}

TEST_CASE("Test transpose, span2d and row operations.", "[test_dot_product]")
{
  CHECK_NOTHROW(test_dot_product());
}

} // namespace