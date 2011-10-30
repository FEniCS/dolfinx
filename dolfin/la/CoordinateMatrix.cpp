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
// First added:  2011-10-15
// Last changed:

#include <vector>
#include "GenericMatrix.h"
#include "CoordinateMatrix.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
CoordinateMatrix::CoordinateMatrix(const GenericMatrix& A, bool base_one)
  : _base_one(base_one)
{
  _size[0] = A.size(0);
  _size[1] = A.size(1);

  // Iterate over local rows
  std::pair<uint, uint> local_row_range = A.local_range(0);
  for (uint i = local_row_range.first; i < local_row_range.second; ++i)
  {
    // Get column and value data for row
    std::vector<uint> columns;
    std::vector<double> values;
    A.getrow(i, columns, values);

    // Insert data at end
    _rows.insert(_rows.end(), columns.size(), i);
    _cols.insert(_cols.end(), columns.begin(), columns.end());
    _vals.insert(_vals.end(), values.begin(), values.end());
  }
  assert(_rows.size() == _cols.size());

  // Add 1 for Fortran-style indices
  if (base_one)
  {
    for (uint i = 0; i < _cols.size(); ++i)
    {
      _rows[i]++;
      _cols[i]++;
    }
  }
}
//-----------------------------------------------------------------------------
