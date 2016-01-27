// Copyright (C) 2008 Kent-Andre Mardal
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
// Modified by Anders Logg, 2008.
// Modified by Garth N. Wells, 2011.
//
// First added:  2008-08-25
// Last changed: 2011-01-22
//

#ifndef __BLOCKVECTOR_H
#define __BLOCKVECTOR_H

#include <string>
#include <vector>
#include <memory>
#include <dolfin/common/types.h>

namespace dolfin
{

  /// Forward declarations
  class GenericVector;

  class BlockVector
  {
  public:

    /// Constructor
    BlockVector(std::size_t n = 0);

    /// Destructor
    virtual ~BlockVector();

    /// Return copy of tensor
    virtual BlockVector* copy() const;

    /// Set function
    void set_block(std::size_t i, std::shared_ptr<GenericVector> v);

    /// Get sub-vector (const)
    std::shared_ptr<const GenericVector> get_block(std::size_t i) const;

    /// Get sub-vector (non-const)
    std::shared_ptr<GenericVector> get_block(std::size_t);

    /// Add multiple of given vector (AXPY operation)
    void axpy(double a, const BlockVector& x);

    /// Return inner product with given vector
    double inner(const BlockVector& x) const;

    /// Return norm of vector
    double norm(std::string norm_type) const;

    /// Return minimum value of vector
    double min() const;

    /// Return maximum value of vector
    double max() const;

    /// Multiply vector by given number
    const BlockVector& operator*= (double a);

    /// Divide vector by given number
    const BlockVector& operator/= (double a);

    /// Add given vector
    const BlockVector& operator+= (const BlockVector& x);

    /// Subtract given vector
    const BlockVector& operator-= (const BlockVector& x);

    /// Assignment operator
    const BlockVector& operator= (const BlockVector& x);

    /// Assignment operator
    const BlockVector& operator= (double a);

    /// Return true if empty
    virtual bool empty() const;

    /// Number of vectors
    std::size_t size() const;

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

  private:

    std::vector<std::shared_ptr<GenericVector>> vectors;

  };

}

#endif
