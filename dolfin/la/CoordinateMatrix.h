// Copyright (C) 2011 Garth N. Wells
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
// First added:  2011-10-16
// Last changed:

#ifndef __DOLFIN_COORDINATE_MATRIX_H
#define __DOLFIN_COORDINATE_MATRIX_H

#include <string>
#include <vector>
#include <dolfin/common/types.h>
#include <dolfin/common/MPI.h>

namespace dolfin
{

  // Forward declarations
  class GenericMatrix;

  /// Coordinate sparse matrix.

  class CoordinateMatrix
  {
  public:

    /// Constructor
    CoordinateMatrix(const GenericMatrix& A, bool symmetric, bool base_one);

    /// Destructor
    virtual ~CoordinateMatrix() {}

    /// Size
    /// @param dim
    /// Dimension (0 or 1)
    std::size_t size(std::size_t dim) const
    { return _size[dim]; }

    /// Get row indices
    const std::vector<std::size_t>& rows() const
    { return _rows; }

    /// Get column indices
    const std::vector<std::size_t>& columns() const
    { return _cols; }

    /// Get values
    const std::vector<double>& values() const
    { return _vals; }

    /// Return norm of matrix
    double norm(std::string norm_type) const;

    /// Get MPI_Comm
    MPI_Comm mpi_comm() const
    { return _mpi_comm; }

    /// Whether indices start from 0 (C-style) or 1 (FORTRAN-style)
    bool base_one() const
    { return _base_one; }

  private:

    // MPI communicator
    MPI_Comm _mpi_comm;

    // Row and column indices
    std::vector<std::size_t> _rows;
    std::vector<std::size_t> _cols;

    // Storage of values
    std::vector<double> _vals;

    // Global size
    std::size_t _size[2];

    // Symmetric storage
    const bool _symmetric;

    // Array base (C/Fortran)
    const bool _base_one;
  };

}

#endif
