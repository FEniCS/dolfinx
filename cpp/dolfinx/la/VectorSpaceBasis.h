// Copyright (C) 2013-2019 Patrick E. Farrell and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Vector.h"
#include <xtl/xspan.hpp>

namespace dolfinx::la
{

/// Orthonormalize a set of vectors
/// @param[in,out] basis The set of vectors to orthonormalise. The
/// vectors must have identical parallel layouts. The vectors are
/// modified in-place.
/// @param[in] tol The tolerance used to detect a linear dependency
template <typename T, typename U>
void orthonormalize(const xtl::span<Vector<T, U>>& basis, double tol = 1.0e-10)
{
  // Loop over each vector in basis
  for (std::size_t i = 0; i < basis.size(); ++i)
  {
    // Orthogonalize vector i with respect to previously orthonormalized
    // vectors
    for (std::size_t j = 0; j < i; ++j)
    {
      // basis_i <- basis_i - dot_ij  basis_j
      T dot_ij = inner_product(basis[i], basis[j]);
      std::transform(basis[j].array().begin(), basis[j].array().end(),
                     basis[i].array().begin(), basis[i].mutable_array().begin(),
                     [dot_ij](auto xj, auto xi) { return xi - dot_ij * xj; });
    }

    // Normalise basis function
    double norm = std::sqrt(inner_product(basis[i], basis[i]));
    std::transform(basis[i].array().begin(), basis[i].array().end(),
                   basis[i].mutable_array().begin(),
                   [norm](auto x) { return x / norm; });
    if (norm < tol)
    {
      throw std::runtime_error(
          "VectorSpaceBasis has linear dependency. Cannot orthogonalize.");
    }
  }
}

/// Test if basis is orthonormal
/// @param[in] basis The set of vectors to check
/// @param[in] tol The tolerance used to test for orthonormality
/// @return True is basis is orthonormal, otherwise false
template <typename T, typename U>
bool is_orthonormal(const xtl::span<const Vector<T, U>>& basis,
                    double tol = 1.0e-10)
{
  for (std::size_t i = 0; i < basis.size(); i++)
  {
    for (std::size_t j = i; j < basis.size(); j++)
    {
      const double delta_ij = (i == j) ? 1.0 : 0.0;
      T dot_ij = inner_product(basis[i], basis[j]);
      if (std::abs(delta_ij - dot_ij) > tol)
        return false;
    }
  }

  return true;
}

} // namespace dolfinx::la
