// Copyright (C) 2021 Igor Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cmath>
#include <type_traits>
#include <xtensor/xfixed.hpp>
#include <xtensor/xtensor.hpp>

namespace dolfinx::math
{

/// Compute the cross product u x v
/// @param u The first vector. It must has size 3.
/// @param v The second vector. It must has size 3.
/// @return The cross product `u x v`. The type will be the same as `u`.
template <typename U, typename V>
xt::xtensor_fixed<typename U::value_type, xt::xshape<3>> cross(const U& u,
                                                               const V& v)
{
  assert(u.size() == 3);
  assert(v.size() == 3);
  return {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
          u[0] * v[1] - u[1] * v[0]};
}

/// Kahan’s method to compute x = ad − bc with fused multiply-adds. The
/// absolute error is bounded by 1.5 ulps, units of least precision.
template <typename T>
T difference_of_products(T a, T b, T c, T d) noexcept
{
  T w = b * c;
  T err = std::fma(-b, c, w);
  T diff = std::fma(a, d, -w);
  return diff + err;
}

/// Compute the determinant of a small matrix (1x1, 2x2, or 3x3)
/// @note Tailored for use in computations using the Jacobian
/// @param[in] A The matrix tp compute the determinant of
/// @return The determinate of @p A
template <typename Matrix>
auto det(const Matrix& A)
{
  using value_type = typename Matrix::value_type;
  assert(A.shape(0) == A.shape(1));
  assert(A.dimension() == 2);

  const int nrows = A.shape(0);
  switch (nrows)
  {
  case 1:
    return A(0, 0);
  case 2:
    return difference_of_products(A(0, 0), A(0, 1), A(1, 0), A(1, 1));
  case 3:
  {
    // Leibniz formula combined with Kahan’s method for accurate
    // computation of 3 x 3 determinants
    value_type w0 = difference_of_products(A(1, 1), A(1, 2), A(2, 1), A(2, 2));
    value_type w1 = difference_of_products(A(1, 0), A(1, 2), A(2, 0), A(2, 2));
    value_type w2 = difference_of_products(A(1, 0), A(1, 1), A(2, 0), A(2, 1));
    value_type w3 = difference_of_products(A(0, 0), A(0, 1), w1, w0);
    value_type w4 = std::fma(A(0, 2), w2, w3);
    return w4;
  }
  default:
    throw std::runtime_error("math::det is not implemented for "
                             + std::to_string(A.shape(0)) + "x"
                             + std::to_string(A.shape(1)) + " matrices.");
  }
}

/// Compute the inverse of a square matrix A (1x1, 2x2 or 3x3) and
/// assign the result to a preallocated matrix B.
/// @param[in] A The matrix to compute the inverse of.
/// @param[out] B The inverse of A. It must be pre-allocated to be the
/// same shape as @p A.
/// @warning This function does not check if A is invertible
template <typename U, typename V>
void inv(const U& A, V&& B)
{
  using value_type = typename U::value_type;
  const std::size_t nrows = A.shape(0);
  switch (nrows)
  {
  case 1:
    B(0, 0) = 1 / A(0, 0);
    break;
  case 2:
  {
    value_type idet = 1. / det(A);
    B(0, 0) = idet * A(1, 1);
    B(0, 1) = -idet * A(0, 1);
    B(1, 0) = -idet * A(1, 0);
    B(1, 1) = idet * A(0, 0);
    break;
  }
  case 3:
  {
    value_type w0 = difference_of_products(A(1, 1), A(1, 2), A(2, 1), A(2, 2));
    value_type w1 = difference_of_products(A(1, 0), A(1, 2), A(2, 0), A(2, 2));
    value_type w2 = difference_of_products(A(1, 0), A(1, 1), A(2, 0), A(2, 1));
    value_type w3 = difference_of_products(A(0, 0), A(0, 1), w1, w0);
    value_type det = std::fma(A(0, 2), w2, w3);
    assert(det != 0.);
    value_type idet = 1 / det;

    B(0, 0) = w0 * idet;
    B(1, 0) = -w1 * idet;
    B(2, 0) = w2 * idet;
    B(0, 1) = difference_of_products(A(0, 2), A(0, 1), A(2, 2), A(2, 1)) * idet;
    B(0, 2) = difference_of_products(A(0, 1), A(0, 2), A(1, 1), A(1, 2)) * idet;
    B(1, 1) = difference_of_products(A(0, 0), A(0, 2), A(2, 0), A(2, 2)) * idet;
    B(1, 2) = difference_of_products(A(1, 0), A(0, 0), A(1, 2), A(0, 2)) * idet;
    B(2, 1) = difference_of_products(A(2, 0), A(0, 0), A(2, 1), A(0, 1)) * idet;
    B(2, 2) = difference_of_products(A(0, 0), A(1, 0), A(0, 1), A(1, 1)) * idet;
    break;
  }
  default:
    throw std::runtime_error("math::inv is not implemented for "
                             + std::to_string(A.shape(0)) + "x"
                             + std::to_string(A.shape(1)) + " matrices.");
  }
}

/// Compute C += A * B
/// @param[in] A Input matrix
/// @param[in] B Input matrix
/// @param[in, out] C Filled to be C += A * B
/// @param[in] transpose Computes C += A^T * B^T if false, otherwise
/// computed C += A^T * B^T
template <typename U, typename V, typename P>
void dot(const U& A, const V& B, P&& C, bool transpose = false)
{
  if (transpose)
  {
    assert(A.shape(0) == B.shape(1));
    for (std::size_t i = 0; i < A.shape(1); i++)
      for (std::size_t j = 0; j < B.shape(0); j++)
        for (std::size_t k = 0; k < A.shape(0); k++)
          C(i, j) += A(k, i) * B(j, k);
  }
  else
  {
    assert(A.shape(1) == B.shape(0));
    for (std::size_t i = 0; i < A.shape(0); i++)
      for (std::size_t j = 0; j < B.shape(1); j++)
        for (std::size_t k = 0; k < A.shape(1); k++)
          C(i, j) += A(i, k) * B(k, j);
  }
}

/// Compute the left pseudo inverse of a rectangular matrix A (3x2, 3x1,
/// or 2x1), such that pinv(A) * A = I
/// @param[in] A The matrix to the compute the pseudo inverse of.
/// @param[out] P The pseudo inverse of @p A. It must be pre-allocated
/// with a size which is the transpose of the size of @p A.
/// @warning The matrix @p A should be full rank
template <typename U, typename V>
void pinv(const U& A, V&& P)
{
  auto shape = A.shape();
  assert(shape[0] > shape[1]);
  assert(P.shape(1) == shape[0]);
  assert(P.shape(0) == shape[1]);
  using T = typename U::value_type;
  if (shape[1] == 2)
  {
    xt::xtensor_fixed<T, xt::xshape<2, 3>> AT;
    xt::xtensor_fixed<T, xt::xshape<2, 2>> ATA;
    xt::xtensor_fixed<T, xt::xshape<2, 2>> Inv;
    AT = xt::transpose(A);
    ATA.fill(0);

    // pinv(A) = (A^T * A)^-1 * A^T
    dolfinx::math::dot(AT, A, ATA);
    dolfinx::math::inv(ATA, Inv);
    dolfinx::math::dot(Inv, AT, P);
  }
  else if (shape[1] == 1)
  {
    auto res = std::transform_reduce(A.begin(), A.end(), 0., std::plus<T>(),
                                     [](const auto v) { return v * v; });
    P = (1 / res) * xt::transpose(A);
  }
  else
  {
    throw std::runtime_error("math::pinv is not implemented for "
                             + std::to_string(A.shape(0)) + "x"
                             + std::to_string(A.shape(1)) + " matrices.");
  }
}

} // namespace dolfinx::math
