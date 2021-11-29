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

/// This class defines a basis for vector spaces, typically used for
/// expressing nullspaces of singular operators and 'near nullspaces'
/// used in smoothed aggregation algebraic multigrid.

template <typename T>
class VectorSpaceBasis
{
public:
  /// Constructor
  template <typename V>
  VectorSpaceBasis(V&& basis) : _basis(std::forward<V>(basis))
  {
    // Do nothing
  }

  /// Delete copy constructor
  VectorSpaceBasis(const VectorSpaceBasis& basis) = delete;

  /// Move constructor
  VectorSpaceBasis(VectorSpaceBasis&& basis) = default;

  /// Destructor
  ~VectorSpaceBasis() = default;

  /// Apply the Gram-Schmidt process to orthonormalize the basis. Throws
  /// an error if a (near) linear dependency is detected. Error is
  /// thrown if <x_i, x_i> < tol.
  void orthonormalize(double tol = 1.0e-10)
  {
    // Loop over each vector in basis
    for (std::size_t i = 0; i < _basis.size(); ++i)
    {
      // Orthogonalize vector i with respect to previously orthonormalized
      // vectors
      for (std::size_t j = 0; j < i; ++j)
      {
        double  dot_ij = inner(_basis[i], _basis[j] );
        // VecDot(_basis[i]->vec(), _basis[j]->vec(), &dot_ij);
        VecAXPY(_basis[i]->vec(), -dot_ij, _basis[j]->vec());
      }

      // Normalise basis function
      PetscReal norm = 0.0;
      VecNormalize(_basis[i]->vec(), &norm);
      if (norm < tol)
      {
        throw std::runtime_error(
            "VectorSpaceBasis has linear dependency. Cannot orthogonalize.");
      }
    }
  }

  /// Test if basis is orthonormal
  bool is_orthonormal(double /*tol = 1.0e-10*/) const
  {
    // for (std::size_t i = 0; i < _basis.size(); i++)
    // {
    //   for (std::size_t j = i; j < _basis.size(); j++)
    //   {
    //     assert(_basis[i]);
    //     assert(_basis[j]);
    //     const double delta_ij = (i == j) ? 1.0 : 0.0;
    //     PetscScalar dot_ij = 0.0;
    //     VecDot(_basis[i]->vec(), _basis[j]->vec(), &dot_ij);

    //     if (std::abs(delta_ij - dot_ij) > tol)
    //       return false;
    //   }
    // }

    return true;
  }

  /// Test if basis is orthogonal
  bool is_orthogonal(double /*tol = 1.0e-10 */) const
  {
    // for (std::size_t i = 0; i < _basis.size(); i++)
    // {
    //   for (std::size_t j = i; j < _basis.size(); j++)
    //   {
    //     assert(_basis[i]);
    //     assert(_basis[j]);
    //     if (i != j)
    //     {
    //       PetscScalar dot_ij = 0.0;
    //       VecDot(_basis[i]->vec(), _basis[j]->vec(), &dot_ij);
    //       if (std::abs(dot_ij) > tol)
    //         return false;
    //     }
    //   }
    // }

    return true;
  }

  /// Test if basis is in null space of A
  // bool in_nullspace(const Mat A, double tol = 1.0e-10) const;

  /// Orthogonalize x with respect to basis
  // void orthogonalize(PETScVector& x) const;

  /// Number of vectors in the basis
  int dim() const { return _basis.size(); }

  /// Get a particular basis vector
  const la::Vector<T>& operator[](int i) const { return _basis.at(i); }

private:
  std::vector<la::Vector<T>> _basis;
};
} // namespace dolfinx::la
