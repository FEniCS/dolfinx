// Copyright (C) 2021 Igor Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <basix/mdspan.hpp>
#include <cmath>
#include <string>
#include <type_traits>

namespace dolfinx::math
{

/// Compute the cross product u x v
/// @param u The first vector. It must has size 3.
/// @param v The second vector. It must has size 3.
/// @return The cross product `u x v`. The type will be the same as `u`.
template <typename U, typename V>
std::array<typename U::value_type, 3> cross(const U& u, const V& v)
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
/// @param[in] A The matrix to compute the determinant of. Row-major
/// storage.
/// @param[in] shape The shape of `A`
/// @return The determinate of `A`
template <typename T>
auto det(const T* A, std::array<std::size_t, 2> shape)
{
  assert(shape[0] == shape[1]);

  // const int nrows = shape[0];
  switch (shape[0])
  {
  case 1:
    return *A;
  case 2:
    /* A(0, 0), A(0, 1), A(1, 0), A(1, 1) */
    return difference_of_products(A[0], A[1], A[2], A[3]);
  case 3:
  {
    // Leibniz formula combined with Kahan’s method for accurate
    // computation of 3 x 3 determinants
    T w0 = difference_of_products(A[3 + 1], A[3 + 2], A[3 * 2 + 1],
                                  A[2 * 3 + 2]);
    T w1 = difference_of_products(A[3], A[3 + 2], A[3 * 2], A[3 * 2 + 2]);
    T w2 = difference_of_products(A[3], A[3 + 1], A[3 * 2], A[3 * 2 + 1]);
    T w3 = difference_of_products(A[0], A[1], w1, w0);
    T w4 = std::fma(A[2], w2, w3);
    return w4;
  }
  default:
    throw std::runtime_error("math::det is not implemented for "
                             + std::to_string(A[0]) + "x" + std::to_string(A[1])
                             + " matrices.");
  }
}

/// Compute the determinant of a small matrix (1x1, 2x2, or 3x3)
/// @note Tailored for use in computations using the Jacobian
/// @param[in] A The matrix tp compute the determinant of
/// @return The determinate of @p A
template <typename Matrix>
auto det(Matrix A)
{
  static_assert(Matrix::rank() == 2, "Must be rank 2");
  assert(A.extent(0) == A.extent(1));

  using value_type = typename Matrix::value_type;
  const int nrows = A.extent(0);
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
                             + std::to_string(A.extent(0)) + "x"
                             + std::to_string(A.extent(1)) + " matrices.");
  }
}

/// Compute the inverse of a square matrix A (1x1, 2x2 or 3x3) and
/// assign the result to a preallocated matrix B.
/// @param[in] A The matrix to compute the inverse of.
/// @param[out] B The inverse of A. It must be pre-allocated to be the
/// same shape as @p A.
/// @warning This function does not check if A is invertible
template <typename U, typename V>
void inv(U A, V B)
{
  static_assert(U::rank() == 2, "Must be rank 2");
  static_assert(V::rank() == 2, "Must be rank 2");

  using value_type = typename U::value_type;
  const std::size_t nrows = A.extent(0);
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
                             + std::to_string(A.extent(0)) + "x"
                             + std::to_string(A.extent(1)) + " matrices.");
  }
}

/// Compute C += A * B
/// @param[in] A Input matrix
/// @param[in] B Input matrix
/// @param[in, out] C Filled to be C += A * B
/// @param[in] transpose Computes C += A^T * B^T if false, otherwise
/// computed C += A^T * B^T
template <typename U, typename V, typename P>
void dot(U A, V B, P C, bool transpose = false)
{
  static_assert(U::rank() == 2, "Must be rank 2");
  static_assert(V::rank() == 2, "Must be rank 2");
  static_assert(P::rank() == 2, "Must be rank 2");

  if (transpose)
  {
    assert(A.extent(0) == B.extent(1));
    for (std::size_t i = 0; i < A.extent(1); i++)
      for (std::size_t j = 0; j < B.extent(0); j++)
        for (std::size_t k = 0; k < A.extent(0); k++)
          C(i, j) += A(k, i) * B(j, k);
  }
  else
  {
    assert(A.extent(1) == B.extent(0));
    for (std::size_t i = 0; i < A.extent(0); i++)
      for (std::size_t j = 0; j < B.extent(1); j++)
        for (std::size_t k = 0; k < A.extent(1); k++)
          C(i, j) += A(i, k) * B(k, j);
  }
}

/// @brief Compute the left pseudo inverse of a rectangular matrix `A`
/// (3x2, 3x1, or 2x1), such that pinv(A) * A = I.
/// @param[in] A The matrix to the compute the pseudo inverse of.
/// @param[out] P The pseudo inverse of `A`. It must be pre-allocated
/// with a size which is the transpose of the size of `A`.
/// @pre The matrix `A` must be full rank
template <typename U, typename V>
void pinv(U A, V P)
{
  static_assert(U::rank() == 2, "Must be rank 2");
  static_assert(V::rank() == 2, "Must be rank 2");

  assert(A.extent(0) > A.extent(1));
  assert(P.extent(1) == A.extent(0));
  assert(P.extent(0) == A.extent(1));
  using T = typename U::value_type;
  if (A.extent(1) == 2)
  {
    namespace stdex = std::experimental;
    std::array<T, 6> ATb;
    std::array<T, 4> ATAb, Invb;
    stdex::mdspan<T, stdex::extents<std::size_t, 2, 3>> AT(ATb.data(), 2, 3);
    stdex::mdspan<T, stdex::extents<std::size_t, 2, 2>> ATA(ATAb.data(), 2, 2);
    stdex::mdspan<T, stdex::extents<std::size_t, 2, 2>> Inv(Invb.data(), 2, 2);

    for (std::size_t i = 0; i < AT.extent(0); ++i)
      for (std::size_t j = 0; j < AT.extent(1); ++j)
        AT(i, j) = A(j, i);

    std::fill(ATAb.begin(), ATAb.end(), 0.0);
    for (std::size_t i = 0; i < P.extent(0); ++i)
      for (std::size_t j = 0; j < P.extent(1); ++j)
        P(i, j) = 0;

    // pinv(A) = (A^T * A)^-1 * A^T
    dot(AT, A, ATA);
    inv(ATA, Inv);
    dot(Inv, AT, P);
  }
  else if (A.extent(1) == 1)
  {
    T res = 0;
    for (std::size_t i = 0; i < A.extent(0); ++i)
      for (std::size_t j = 0; j < A.extent(1); ++j)
        res += A(i, j) * A(i, j);

    for (std::size_t i = 0; i < A.extent(0); ++i)
      for (std::size_t j = 0; j < A.extent(1); ++j)
        P(j, i) = (1 / res) * A(i, j);
  }
  else
  {
    throw std::runtime_error("math::pinv is not implemented for "
                             + std::to_string(A.extent(0)) + "x"
                             + std::to_string(A.extent(1)) + " matrices.");
  }
}

} // namespace dolfinx::math
