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

#include <cmath>
#include <vector>
#include "dolfin/common/MPI.h"
#include "GenericMatrix.h"
#include "CoordinateMatrix.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
CoordinateMatrix::CoordinateMatrix(const GenericMatrix& A, bool symmetric,
                                   bool base_one)
  : _mpi_comm(A.mpi_comm()), _symmetric(symmetric), _base_one(base_one)
{
  _size[0] = A.size(0);
  _size[1] = A.size(1);

  // Iterate over local rows
  const std::pair<std::size_t, std::size_t> local_row_range = A.local_range(0);
  if (!_symmetric)
  {
    for (std::size_t i = local_row_range.first; i < local_row_range.second; ++i)
    {
      // Get column and value data for row
      std::vector<std::size_t> columns;
      std::vector<double> values;
      A.getrow(i, columns, values);

      // Insert data at end
      _rows.insert(_rows.end(), columns.size(), i);
      _cols.insert(_cols.end(), columns.begin(), columns.end());
      _vals.insert(_vals.end(), values.begin(), values.end());
    }
    dolfin_assert(_rows.size() == _cols.size());
  }
  else
  {
    dolfin_assert(_size[0] == _size[1]);
    for (std::size_t i = local_row_range.first; i < local_row_range.second; ++i)
    {
      // Get column and value data for row
      std::vector<std::size_t> columns;
      std::vector<double> values;
      A.getrow(i, columns, values);

      for (std::size_t j = 0; j < columns.size(); ++j)
      {
        if (columns[j] >= i)
        {
          _rows.push_back(i);
          _cols.push_back(columns[j]);
          _vals.push_back(values[j]);
        }
      }
    }
    dolfin_assert(_rows.size() == _cols.size());
  }

  // Add 1 for Fortran-style indices
  if (base_one)
  {
    for (std::size_t i = 0; i < _cols.size(); ++i)
    {
      _rows[i]++;
      _cols[i]++;
    }
  }
}
//-----------------------------------------------------------------------------
double CoordinateMatrix::norm(std::string norm_type) const
{
  if (norm_type != "frobenius")
  {
    dolfin_error("CoordinateMatrix.cpp",
                 "compute matrix norm",
                 "Do not know to compute %s norm for CoordinateMatrix",
                 norm_type.c_str());
  }

  double _norm = 0.0;
  if (!_symmetric)
  {
    for (std::size_t i = 0; i < _vals.size(); ++i)
      _norm += _vals[i]*_vals[i];
  }
  else
  {
    for (std::size_t i = 0; i < _vals.size(); ++i)
    {
      if (_rows[i] == _cols[i])
        _norm += _vals[i]*_vals[i];
      else
        _norm += 2.0*_vals[i]*_vals[i];
    }
  }
  return std::sqrt(dolfin::MPI::sum(_mpi_comm, _norm));
}
//-----------------------------------------------------------------------------
