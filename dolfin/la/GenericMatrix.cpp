// Copyright (C) 2010 Anders Logg
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
//
// First added:  2010-02-23
// Last changed: 2011-03-17

#include <boost/scoped_array.hpp>
#include <dolfin/common/constants.h>
#include "GenericMatrix.h"
#include "GenericSparsityPattern.h"
#include <dolfin/common/Timer.h>
#include <dolfin/la/LinearAlgebraFactory.h>

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
  
  std::vector<uint> columns;
  std::vector<double> values;
  std::vector<uint> zero_rows;
  const std::pair<uint, uint> row_range = local_range(0);
  const uint m = row_range.second - row_range.first;

  // Check which rows are zero
  for (uint row = 0; row < m; row++)
  {
    // Get global row number
    int global_row = row + row_range.first;
    
    // Get value for row
    getrow(global_row, columns, values);

    // Get maximum value in row
    double max = 0.0;
    for (uint k = 0; k < values.size(); k++)
      max = std::max(max, std::abs(values[k]));

    // Check if row is zero
    if (max < DOLFIN_EPS)
      zero_rows.push_back(global_row);
  }

  // Write a message
  log(TRACE, "Found %d zero row(s), inserting ones on the diagonal.", zero_rows.size());

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

void GenericMatrix::compress()
{
  Timer timer("Compress matrix");
  
  // Create new sparsity pattern
  boost::shared_ptr<GenericSparsityPattern> new_sparsity_pattern = factory().create_pattern();
    
  // Retrieve global and local matrix info
  std::vector<uint> global_dimensions(2);
  global_dimensions[0] = size(0);
  global_dimensions[1] = size(1);
  std::vector<std::pair<uint, uint> > loc_range(2);
  loc_range[0] = local_range(0);
  loc_range[1].first = 0;  // Column range not provided by all backends
  loc_range[1].second = size(1);
  
  // With the row-by-row algorithm used here there is no need for inserting non_local 
  // rows and as such we can simply use a dummy for off_process_owner
  std::vector<const boost::unordered_map<uint, uint>* > off_process_owner(2);
  const boost::unordered_map<uint, uint> dummy;
  off_process_owner[0] = &dummy;
  off_process_owner[1] = &dummy;
  const std::pair<uint, uint> row_range = loc_range[0];
  const std::pair<uint, uint> col_range = loc_range[1];
  const uint m = row_range.second - row_range.first;
    
  // Initialize sparsity pattern
  new_sparsity_pattern->init(global_dimensions, loc_range, off_process_owner);
    
  // Declare some variables used to extract matrix information
  std::vector<uint> columns;
  std::vector<double> values;
  std::vector<double> allvalues;   // Hold all values of local matrix
  std::vector<uint> allcolumns;    // Hold the column id for all values of local matrix
  std::vector<uint> offset(m+1);   // Hold accumulated number of cols on local matrix
  offset[0] = 0;
  std::vector<uint> thisrow(1);
  std::vector<uint> thiscolumn;
  std::vector<const std::vector<uint>* > dofs(2);
  dofs[0] = &thisrow;
  dofs[1] = &thiscolumn;

  for (uint i = 0; i < m; i++)
  {
    // Get row and locate nonzeros. Store non-zero values and columns for later
    int global_row = i + row_range.first;        
    getrow(global_row, columns, values);
    uint count = 0;
    thiscolumn.clear();
    for (uint j = 0; j < columns.size(); j++)
    {
      // Store if non-zero or diagonal entry. PETSc solvers require this
      if(std::abs(values[j]) > DOLFIN_EPS || columns[j] == global_row)
      {
        thiscolumn.push_back(columns[j]);
        allvalues.push_back(values[j]);
        allcolumns.push_back(columns[j]);
        count++;
      }
    }
    
    thisrow[0] = global_row;
    offset[i+1] = offset[i] + count;

    // Build new compressed sparsity pattern
    new_sparsity_pattern->insert(dofs);
  }
  
  // Finalize sparsity pattern
  new_sparsity_pattern->apply();

  // Recreate matrix with the new sparsity pattern
  init(*new_sparsity_pattern);
    
  // Put the old values back in the newly compressed matrix
  for (uint i = 0; i < m; i++)
  {
    uint global_row = i + row_range.first;
    set(&allvalues[offset[i]], 1, &global_row, offset[i+1]-offset[i], &allcolumns[offset[i]]);
  }

  apply("insert");      
}
//-----------------------------------------------------------------------------
