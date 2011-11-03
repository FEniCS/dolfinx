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

    unsigned int size(unsigned int dim) const
    { return _size[dim]; }

    const std::vector<uint>& rows() const
    { return _rows; }

    const std::vector<uint>& columns() const
    { return _cols; }

    const std::vector<double>& values() const
    { return _vals; }

    /// Return norm of matrix
    double norm(std::string norm_type) const;

    bool base_one() const
    { return _base_one; }

  private:

    // Row and column indices
    std::vector<uint> _rows;
    std::vector<uint> _cols;

    // Storage of values
    std::vector<double> _vals;

    // Gobal size
    unsigned int _size[2];

    // Symmetric storage
    const bool _symmetric;

    // Array base (C/Fortran)
    const bool _base_one;
  };

}

#endif
