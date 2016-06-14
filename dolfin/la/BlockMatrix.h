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

#ifndef __BLOCKMATRIX_H
#define __BLOCKMATRIX_H

#include <boost/multi_array.hpp>
#include <memory>

namespace dolfin
{

  // Forward declarations
  class GenericMatrix;

  /// Block Matrix

  class BlockMatrix
  {
  public:

    // Constructor
    BlockMatrix(std::size_t m=0, std::size_t n=0);

    // Destructor
    ~BlockMatrix();

    /// Set block
    void set_block(std::size_t i, std::size_t j,
                   std::shared_ptr<GenericMatrix> m);

    /// Get block (const version)
    std::shared_ptr<const GenericMatrix>
      get_block(std::size_t i, std::size_t j) const;

    /// Get block
    std::shared_ptr<GenericMatrix> get_block(std::size_t i, std::size_t j);

    /// Return size of given dimension
    std::size_t size(std::size_t dim) const;

    /// Set all entries to zero and keep any sparse structure
    void zero();

    /// Finalize assembly of tensor
    void apply(std::string mode);

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Matrix-vector product, y = Ax
    void mult(const BlockVector& x, BlockVector& y,
              bool transposed=false) const;

    /// Create a crude explicit Schur approximation  of S = D - C A^-1
    /// B of  (A B; C  D) If symmetry !=  0, then the  caller promises
    /// that B = symmetry * transpose(C).
    std::shared_ptr<GenericMatrix>
      schur_approximation(bool symmetry=true) const;

  private:

    boost::multi_array<std::shared_ptr<GenericMatrix>, 2> matrices;

  };

}

#endif
