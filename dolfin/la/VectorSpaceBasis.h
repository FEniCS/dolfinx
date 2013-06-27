// Copyright (C) 2013 Patrick E. Farrell
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
//
// First added:  2013-05-29
// Last changed: 2013-05-29

#ifndef __VECTOR_SPACE_BASIS_H
#define __VECTOR_SPACE_BASIS_H

#include <vector>
#include <boost/shared_ptr.hpp>
#include "GenericVector.h"

namespace dolfin
{

  /// This class defines a basis for vector spaces,
  /// typically used for expressing nullspaces, transpose nullspaces
  /// and near nullspaces of singular operators

  class VectorSpaceBasis
  {
  public:

    /// Constructor
    VectorSpaceBasis(const std::vector<boost::shared_ptr<GenericVector> > basis);

    /// Destructor
    ~VectorSpaceBasis() {}

    /// Test if basis is orthonormal
    bool is_orthonormal() const;

    /// Test if basis is orthogonal
    bool is_orthogonal() const;

    /// Orthogonalize x with respect to basis
    void orthogonalize(GenericVector& x) const;

    /// Dimension of the basis
    std::size_t dim() const;

    /// Get a particular basis vector
    boost::shared_ptr<const GenericVector> operator[] (std::size_t i) const;

  private:

    // Basis vectors
    const std::vector<boost::shared_ptr<GenericVector> > _basis;

  };
}

#endif
