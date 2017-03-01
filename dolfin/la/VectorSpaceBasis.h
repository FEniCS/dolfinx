// Copyright (C) 2013-2017 Patrick E. Farrell and Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

#ifndef __VECTOR_SPACE_BASIS_H
#define __VECTOR_SPACE_BASIS_H

#include <memory>
#include <vector>

namespace dolfin
{

  class GenericVector;

  /// This class defines a basis for vector spaces, typically used for
  /// expressing nullspaces of singular operators and 'near
  /// nullspaces' used in smoothed aggregation algebraic multigrid.

  class VectorSpaceBasis
  {
  public:

    /// Constructor
    VectorSpaceBasis(const std::vector<std::shared_ptr<GenericVector>> basis);

    /// Destructor
    ~VectorSpaceBasis() {}

    /// Apply the Gram-Schmidt process to orthonormalize the
    /// basis. Throws an error if a (near) linear dependency is
    /// detected. Error is thrown if <x_i, x_i> < tol.
    void orthonormalize(double tol=1.0e-10);

    /// Test if basis is orthonormal
    bool is_orthonormal(double tol=1.0e-10) const;

    /// Test if basis is orthogonal
    bool is_orthogonal(double tol=1.0e-10) const;

    /// Orthogonalize x with respect to basis
    void orthogonalize(GenericVector& x) const;

    /// Number of vectors in the basis
    std::size_t dim() const;

    /// Get a particular basis vector
    std::shared_ptr<const GenericVector> operator[] (std::size_t i) const;

  private:

    // Basis vectors
    const std::vector<std::shared_ptr<GenericVector>> _basis;

  };
}

#endif
