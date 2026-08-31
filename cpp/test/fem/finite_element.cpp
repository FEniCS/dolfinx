// Copyright (C) 2026 Paul T. Kuehner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <basix/finite-element.h>

#include <dolfinx/fem/FiniteElement.h>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <span>
#include <string_view>
#include <vector>

using namespace dolfinx;

namespace
{
using Func = std::function<void(
    std::span<double>, std::span<const std::uint32_t>, std::int32_t, int)>;

/// The sub-span slicing under test is shared by every transformation
/// type, so all four are exercised.
constexpr std::array ttypes{
    fem::doftransform::standard, fem::doftransform::transpose,
    fem::doftransform::inverse, fem::doftransform::inverse_transpose};

/// Transformation type name, for test failure output.
std::string_view name(fem::doftransform ttype)
{
  switch (ttype)
  {
  case fem::doftransform::standard:
    return "standard";
  case fem::doftransform::transpose:
    return "transpose";
  case fem::doftransform::inverse:
    return "inverse";
  case fem::doftransform::inverse_transpose:
    return "inverse_transpose";
  default:
    return "unknown";
  }
}

/// Nedelec (first kind), degree 2, on a triangle. Two DOFs per edge, so
/// the DOF transformations are non-trivial (and not permutations).
basix::FiniteElement<double> nedelec()
{
  return basix::create_element<double>(
      basix::element::family::N1E, basix::cell::type::triangle, 2,
      basix::element::lagrange_variant::legendre,
      basix::element::dpc_variant::unset, false);
}

/// P1 Lagrange on a triangle. DOF transformations are the identity, so
/// the only role of this sub-element is to shift the offset of the
/// element it is mixed with.
basix::FiniteElement<double> lagrange()
{
  return basix::create_element<double>(
      basix::element::family::P, basix::cell::type::triangle, 1,
      basix::element::lagrange_variant::gll_isaac,
      basix::element::dpc_variant::unset, false);
}

/// Apply `fn` to `A` (shape `(nrows, ncols)`, row-major) one row at a
/// time, i.e. exclusively through the `block_size == 1` code path.
std::vector<double> apply_row_by_row(const Func& fn, std::span<const double> A,
                                     int nrows, int ncols,
                                     std::span<const std::uint32_t> cell_info)
{
  std::vector<double> out(A.begin(), A.end());
  for (int i = 0; i < nrows; ++i)
  {
    const std::uint32_t cell = 0;
    const int block_size = 1;
    fn(std::span(out).subspan(i * ncols, ncols), cell_info, cell, block_size);
  }
  return out;
}

/// Row-major matrix of distinct, well-separated entries.
std::vector<double> test_data(int nrows, int ncols)
{
  std::vector<double> A(nrows * ncols);
  std::iota(A.begin(), A.end(), 1.0);
  return A;
}
} // namespace

TEST_CASE("Mixed element DOF transformation from the right, block size > 1",
          "[fem][finite_element]")
{
  // Cell permutation with edges 0 and 1 reflected.
  const std::array<std::uint32_t, 1> cell_info{0b011};

  const auto sub_nedelec
      = std::make_shared<const fem::FiniteElement<double>>(nedelec());
  const auto sub_lagrange
      = std::make_shared<const fem::FiniteElement<double>>(lagrange());

  // The transforming sub-element is placed second, so it sits at a
  // non-zero DOF offset within the mixed element.
  const fem::FiniteElement<double> e({sub_lagrange, sub_nedelec});
  REQUIRE(e.needs_dof_transformations());

  const fem::doftransform ttype = GENERATE(from_range(ttypes));
  CAPTURE(name(ttype));

  const int ncols = e.space_dimension();
  const Func Pt = e.dof_transformation_right_fn<double>(ttype);
  REQUIRE(Pt);

  SECTION("Single row is unaffected")
  {
    // The block_size == 1 path has no rows to stride over, so it is
    // taken as the reference below.
    const std::vector<double> A = test_data(1, ncols);
    std::vector<double> B = A;
    Pt(B, cell_info, /*cell=*/0, /*block_size=*/1);
    CHECK(A != B); // Transformation is not a no-op for this cell
  }

  SECTION("Multiple rows match a row-by-row application")
  {
    const int nrows = 3;
    const std::vector<double> A = test_data(nrows, ncols);
    const std::vector<double> expected
        = apply_row_by_row(Pt, A, nrows, ncols, cell_info);

    std::vector<double> B = A;
    Pt(B, cell_info, /*cell=*/0, nrows);
    CHECK(B == expected);
  }
}

TEST_CASE("Mixed element DOF transformation from the right, zero offset",
          "[fem][finite_element]")
{
  // Same as above, but with the transforming sub-element first, i.e. at
  // offset 0. This is the case that the sub-span slicing gets right.
  const std::array<std::uint32_t, 1> cell_info{0b011};

  const auto sub_nedelec
      = std::make_shared<const fem::FiniteElement<double>>(nedelec());
  const auto sub_lagrange
      = std::make_shared<const fem::FiniteElement<double>>(lagrange());

  const fem::FiniteElement<double> e({sub_nedelec, sub_lagrange});
  REQUIRE(e.needs_dof_transformations());

  const fem::doftransform ttype = GENERATE(from_range(ttypes));
  CAPTURE(name(ttype));

  const int ncols = e.space_dimension();
  const Func Pt = e.dof_transformation_right_fn<double>(ttype);
  REQUIRE(Pt);

  const int nrows = 3;
  const std::vector<double> A = test_data(nrows, ncols);
  const std::vector<double> expected
      = apply_row_by_row(Pt, A, nrows, ncols, cell_info);

  std::vector<double> B = A;
  Pt(B, cell_info, /*cell=*/0, nrows);
  CHECK(B == expected);
}

TEST_CASE("Non-mixed element DOF transformation from the right, block size > 1",
          "[fem][finite_element]")
{
  // The leaf (non-mixed) code path with several rows, for contrast with
  // the mixed cases above.
  const std::array<std::uint32_t, 1> cell_info{0b011};

  const fem::FiniteElement<double> e(nedelec());
  REQUIRE(e.needs_dof_transformations());

  const fem::doftransform ttype = GENERATE(from_range(ttypes));
  CAPTURE(name(ttype));

  const int ncols = e.space_dimension();
  const Func Pt = e.dof_transformation_right_fn<double>(ttype);
  REQUIRE(Pt);

  const int nrows = 3;
  const std::vector<double> A = test_data(nrows, ncols);
  const std::vector<double> expected
      = apply_row_by_row(Pt, A, nrows, ncols, cell_info);

  std::vector<double> B = A;
  Pt(B, cell_info, /*cell=*/0, nrows);
  CHECK(B == expected);
}
