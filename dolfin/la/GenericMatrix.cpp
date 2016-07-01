// Copyright (C) 2010-2012 Anders Logg
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
// Modified by Mikael Mortensen 2011

#include <cmath>
#include <vector>

#include <dolfin/common/constants.h>
#include <dolfin/common/Timer.h>
#include "GenericMatrix.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void GenericMatrix::ident_zeros()
{
  // Check size of system
  if (size(0) != size(1))
  {
    dolfin_error("GenericMatrix.cpp",
                 "ident_zeros",
                 "Matrix is not square");
  }

  std::vector<std::size_t> columns;
  std::vector<double> values;
  std::vector<dolfin::la_index> zero_rows;
  const std::pair<std::int64_t, std::int64_t> row_range = local_range(0);
  const std::size_t m = row_range.second - row_range.first;

  // Check which rows are zero
  for (std::size_t row = 0; row < m; row++)
  {
    // Get global row number
    int global_row = row + row_range.first;

    // Get value for row
    getrow(global_row, columns, values);

    // Get maximum value in row
    double max = 0.0;
    for (std::size_t k = 0; k < values.size(); k++)
      max = std::max(max, std::abs(values[k]));

    // Check if row is zero
    if (max < DOLFIN_EPS)
      zero_rows.push_back(global_row);
  }

  // Write a message
  log(TRACE, "Found %d zero row(s), inserting ones on the diagonal.",
      zero_rows.size());

  // Insert one on the diagonal for rows with only zeros.
  ident(zero_rows.size(), zero_rows.data());

  // Apply changes
  apply("insert");
}
//-----------------------------------------------------------------------------
