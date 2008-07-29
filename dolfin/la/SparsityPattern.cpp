// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Magnus Vikstrom, 2008.
// Modified by Anders Logg, 2008.
//
// First added:  2007-03-13
// Last changed: 2008-05-15

#include <dolfin/log/dolfin_log.h>
#include "SparsityPattern.h"
#include <dolfin/main/MPI.h>
//#include <dolfin/PETScObject.h>
#include <iostream>

using namespace dolfin;

//-----------------------------------------------------------------------------
SparsityPattern::SparsityPattern(uint M, uint N) : range(0)
{
  uint dims[2];
  dims[0] = M;
  dims[1] = N;
  init(2, dims);
}
//-----------------------------------------------------------------------------
SparsityPattern::SparsityPattern(uint M) : range(0)
{
  uint dims[2];
  dims[0] = M;
  dims[1] = 0;
  init(1, dims);
}
//-----------------------------------------------------------------------------
  SparsityPattern::SparsityPattern() : range(0)
{
  dim[0] = 0;
  dim[1] = 0;
  sparsity_pattern.clear();
  o_sparsity_pattern.clear();
}
//-----------------------------------------------------------------------------
SparsityPattern::~SparsityPattern()
{
  if(range)
    delete [] range;
}
//-----------------------------------------------------------------------------
void SparsityPattern::init(uint rank, const uint* dims)
{
  dolfin_assert(rank <= 2);
  dim[0] = dim[1] = 0;
  for (uint i = 0; i < rank; ++i)
    dim[i] = dims[i];
  sparsity_pattern.clear();
  sparsity_pattern.resize(dim[0]);
}
//-----------------------------------------------------------------------------
void SparsityPattern::pinit(uint rank, const uint* dims)
{
  dolfin_assert(rank <= 2);
  dim[0] = dim[1] = 0;
  for (uint i = 0; i < rank; ++i)
    dim[i] = dims[i];
  sparsity_pattern.clear();
  sparsity_pattern.resize(dim[0]);
  o_sparsity_pattern.clear();
  o_sparsity_pattern.resize(dim[0]);
  initRange();
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert(uint m, const uint* rows, uint n, const uint* cols)
{ 
  for (uint i = 0; i < m; ++i)
    for (uint j = 0; j < n; ++j)
    {
      bool inserted = false;
      uint k = 0;
      while(k < sparsity_pattern[rows[i]].size() && !inserted)
      {
        if(cols[j] == sparsity_pattern[rows[i]][k])
          inserted = true;
        ++k;
      }
      if(!inserted)
        sparsity_pattern[rows[i]].push_back(cols[j]);
    }

//  for (uint i = 0; i < m; ++i)
//    for (uint j = 0; j < n; ++j)
//      sparsity_pattern[rows[i]].insert(cols[j]);
}
//-----------------------------------------------------------------------------
void SparsityPattern::pinsert(const uint* num_rows, const uint * const * rows)
{ 
  uint process = dolfin::MPI::processNumber();

  for (unsigned int i = 0; i<num_rows[0];++i)
  {
    const uint global_row = rows[0][i];
    // If not in a row "owned" by this processor
    if(global_row < range[process] || global_row >= range[process+1])
      continue;
    for (unsigned int j = 0; j<num_rows[1];++j)
    {
      const uint global_col = rows[1][j];
      // On the off-diagonal
      if(global_col < range[process] || global_col >= range[process+1])
        o_sparsity_pattern[rows[0][i]].insert(rows[1][j]);
      // On the diagonal
//      else
//        sparsity_pattern[rows[0][i]].insert(rows[1][j]);
    }
  }
}
//-----------------------------------------------------------------------------
dolfin::uint SparsityPattern::size(uint n) const
{
  dolfin_assert(n < 2);
  return dim[n]; 
}
//-----------------------------------------------------------------------------
void SparsityPattern::numNonZeroPerRow(uint nzrow[]) const
{
  if ( dim[1] == 0 )
    error("Non-zero entries per row can be computed for matrices only.");

  if ( sparsity_pattern.size() == 0 )
    error("Sparsity pattern has not been computed.");

  // Compute number of nonzeros per row
  std::vector< std::vector<uint> >::const_iterator row;
  for(row = sparsity_pattern.begin(); row != sparsity_pattern.end(); ++row)
    nzrow[row - sparsity_pattern.begin()] = row->size();

//  for(uint i=0; i < sparsity_pattern.size(); ++i)
//    cout << "nz " << sparsity_pattern[i].size() << "  " << mysparsity_pattern[i].size() << endl; 
}
//-----------------------------------------------------------------------------
void SparsityPattern::numNonZeroPerRow(uint process_number, uint d_nzrow[], uint o_nzrow[]) const
{
  if ( dim[1] == 0 )
    error("Non-zero entries per row can be computed for matrices only.");

  if ( sparsity_pattern.size() == 0 )
    error("Sparsity pattern has not been computed.");

  // Compute number of nonzeros per row diagonal and off-diagonal
  uint offset = range[process_number];
  for(uint i=0; i+offset<range[process_number+1]; ++i)
  {
    d_nzrow[i] = sparsity_pattern[i+offset].size();
    o_nzrow[i] = o_sparsity_pattern[i+offset].size();
  }
}
//-----------------------------------------------------------------------------
dolfin::uint SparsityPattern::numNonZero() const
{
  if ( dim[1] == 0 )
    error("Total non-zeros entries can be computed for matrices only.");

  if ( sparsity_pattern.size() == 0 )
    error("Sparsity pattern has not been computed.");

  // Compute total number of nonzeros per row
  uint nz = 0;
  std::vector< std::vector<uint> >::const_iterator set;
  for(set = sparsity_pattern.begin(); set != sparsity_pattern.end(); ++set)
    nz += set->size();
  return nz;
}
//-----------------------------------------------------------------------------
void SparsityPattern::disp() const
{ 
  error("SparsityPattern::disp() needs to be updated.");
/*
  if ( dim[1] == 0 )
    warning("Only matrix sparsity patterns can be displayed.");

  std::vector< std::set<int> >::const_iterator set;
  std::set<int>::const_iterator element;
  
  for(set = sparsity_pattern.begin(); set != sparsity_pattern.end(); ++set)
  {
    cout << "Row " << endl;
    for(element = set->begin(); element != set->end(); ++element)
      cout << *element << " ";
    cout << endl;
  }  
*/
}
//-----------------------------------------------------------------------------
void SparsityPattern::processRange(uint process_number, uint local_range[])
{
  local_range[0] = range[process_number];
  local_range[1] = range[process_number + 1];
}
//-----------------------------------------------------------------------------
dolfin::uint SparsityPattern::numLocalRows(uint process_number) const
{
  return range[process_number + 1] - range[process_number];
}
//-----------------------------------------------------------------------------
void SparsityPattern::initRange()
{
  uint num_procs = dolfin::MPI::numProcesses();
  range = new uint[num_procs+1];
  range[0] = 0;

  for(uint p=0; p<num_procs; ++p)
    range[p+1] = range[p] + dim[0]/num_procs + ((dim[0]%num_procs) > p ? 1 : 0);
}
//-----------------------------------------------------------------------------
