// Copyright (C) 2013-2017 Patrick E. Farrell and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <memory>
#include <vector>

namespace dolfin
{
namespace la
{
class PETScVector;

/// This class defines a basis for vector spaces, typically used for
/// expressing nullspaces of singular operators and 'near
/// nullspaces' used in smoothed aggregation algebraic multigrid.

class VectorSpaceBasis
{
public:
  /// Constructor
  VectorSpaceBasis(const std::vector<std::shared_ptr<PETScVector>> basis);

  /// Destructor
  ~VectorSpaceBasis() {}

  /// Apply the Gram-Schmidt process to orthonormalize the
  /// basis. Throws an error if a (near) linear dependency is
  /// detected. Error is thrown if <x_i, x_i> < tol.
  void orthonormalize(double tol = 1.0e-10);

  /// Test if basis is orthonormal
  bool is_orthonormal(double tol = 1.0e-10) const;

  /// Test if basis is orthogonal
  bool is_orthogonal(double tol = 1.0e-10) const;

  /// Orthogonalize x with respect to basis
  void orthogonalize(PETScVector& x) const;

  /// Number of vectors in the basis
  std::size_t dim() const;

  /// Get a particular basis vector
  std::shared_ptr<const PETScVector> operator[](std::size_t i) const;

private:
  // Basis vectors
  const std::vector<std::shared_ptr<PETScVector>> _basis;
};
}
}