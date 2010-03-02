// Copyright (C) 2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-02-23
// Last changed: 2010-02-23

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
  info("Found %d zero row(s), inserting one on the diagonal.", zero_rows.size());

  // Zero rows
  boost::scoped_array<uint> rows(new uint[zero_rows.size()]);
  for (uint k = 0; k < zero_rows.size(); k++)
    rows[k] = zero_rows[k];
  ident(zero_rows.size(), rows.get());
}
//-----------------------------------------------------------------------------
