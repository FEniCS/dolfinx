// Copyright (C) 2021 Igor Baratta
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cmath>

namespace dolfinx::linalg
{

/// Kahan’s method to compute x = ad − bc with fused multiply-adds
/// The absolute error is bounded by 1.5 ulps, units of least precision.
template <typename T>
inline T difference_of_products(T a, T b, T c, T d) noexcept
{
  double w = b * c;
  double err = std::fma(-b, c, w);
  double diff = std::fma(a, d, -w);
  return (diff + err);
}

/// Compute the determinant of a small matrix (1x1, 2x2, or 3x3)
/// Tailored for use in computations usin the Jacobian
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
    throw std::runtime_error("det is no implement for "
                             + std::to_string(A.shape(0)) + "x"
                             + std::to_string(A.shape(0)) + " matrices");
  }
}

/// Compute the inverse of a square matrix A and assign the result to a
/// preallocated matrix B.
/// @warning This function does not check if A is invertible!
template <typename Matrix>
void inv(const Matrix& A, Matrix& B)
{
  using value_type = typename Matrix::value_type;
  value_type idet = 0;
  const int nrows = A.shape(0);
  switch (nrows)
  {
  case 1:
    B(0, 0) = 1 / A(0, 0);
  case 2:
  {
    idet = 1. / compute_determinant(A);
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
    idet = 1 / det;

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
    throw std::runtime_error("det is no implement for "
                             + std::to_string(A.shape(0)) + "x"
                             + std::to_string(A.shape(0)) + " matrices");
  }
}

/// Compute the dot product of two matrices, where the dimensions of the first
/// matrix are know at compile time.
template <std::size_t N, std::size_t M, typename Matrix>
void dot(const Matrix& A, const Matrix& B, Matrix& C)
{
  using value_type = typename Matrix::value_type;
  for (std::size_t i = 0; i < N; i++)
  {
    for (std::size_t j = 0; j < B.shape(1); j++)
    {
      value_type result = 0;
      for (std::size_t k = 0; k < M; k++)
        result += A(i, k) * B(k, j);
      C(i, j) = result;
    }
  }
}

} // namespace dolfinx::linalg