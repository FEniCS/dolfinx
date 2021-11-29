// Copyright (C) 2013-2019 Patrick E. Farrell and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Vector.h"
#include <vector>

namespace dolfinx::la
{

template <typename V>
void orthonormalize(const xtl::span<V>& basis, double tol = 1.0e-10)
{
  // Loop over each vector in basis
  for (std::size_t i = 0; i < basis.size(); ++i)
  {
    // Orthogonalize vector i with respect to previously orthonormalized
    // vectors
    for (std::size_t j = 0; j < i; ++j)
    {
      double dot_ij = inner_product(basis[i], basis[j]);

      // basis_i <- basis_i - dot_ij  basis_j
      std::transform(basis[j].array().begin(), basis[j].array().begin(),
                     basis[i].array().begin(), basis[i].mutable_array().begin(),
                     [dot_ij](auto xi, auto xj) { return xi - dot_ij * xj; });
    }

    // Normalise basis function
    double norm = inner_product(basis[i], basis[i]);
    std::transform(basis[i].array().begin(), basis[i].array().begin(),
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
template <typename V>
bool is_orthonormal(const xtl::span<V>& basis, double tol = 1.0e-10)
{
  for (std::size_t i = 0; i < basis.size(); i++)
  {
    for (std::size_t j = i; j < basis.size(); j++)
    {
      const double delta_ij = (i == j) ? 1.0 : 0.0;
      typename V::value_type dot_ij = inner_product(basis[i], basis[j]);
      if (std::abs(delta_ij - dot_ij) > tol)
        return false;
    }
  }

  return true;
}

} // namespace dolfinx::la
