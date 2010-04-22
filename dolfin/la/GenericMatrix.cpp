// Copyright (C) 2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-02-23
// Last changed: 2010-04-22

#include <boost/scoped_array.hpp>
#include <dolfin/common/constants.h>
#include "GenericMatrix.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void GenericMatrix::ident_zeros()
{
  std::vector<uint> columns;
  std::vector<double> values;
  std::vector<uint> zero_rows;

  // Check which rows are zero
  for (uint row = 0; row < size(0); row++)
  {
    // Get value for row
    getrow(row, columns, values);

    // Get maximum value in row
    double max = 0.0;
    for (uint k = 0; k < values.size(); k++)
      max = std::max(max, std::abs(values[k]));

    // Check if row is zero
    if (max < DOLFIN_EPS)
      zero_rows.push_back(row);
  }

  // Write a message
  info("Found %d zero row(s), inserting ones on the diagonal.", zero_rows.size());

  // Insert one on the diagonal for rows with only zeros. Note that we
  // are not calling ident() since that fails in PETSc if nothing
  // has been assembled into those rows.
  for (uint i = 0; i < zero_rows.size(); i++)
  {
    std::pair<uint, uint> ij(zero_rows[i], zero_rows[i]);
    setitem(ij, 1.0);
  }

  // Apply changes
  apply("insert");
}
//-----------------------------------------------------------------------------
