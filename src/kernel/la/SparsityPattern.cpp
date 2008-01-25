// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Magnus Vikstrom 2008.
//
// First added:  2007-03-13
// Last changed: 2008-01-24

#include <dolfin/dolfin_log.h>
#include <dolfin/SparsityPattern.h>
#include <dolfin/MPI.h>
//#include <dolfin/PETScObject.h>
#include <iostream>

using namespace dolfin;

//-----------------------------------------------------------------------------
SparsityPattern::SparsityPattern() 
{
  dim[0] = 0;
  dim[1] = 0;
  sparsity_pattern.clear();
  o_sparsity_pattern.clear();
}
//-----------------------------------------------------------------------------
SparsityPattern::~SparsityPattern()
{
  //Do nothing
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
void SparsityPattern::insert(const uint* num_rows, const uint * const * rows)
{ 
  for (unsigned int i = 0; i<num_rows[0];++i)
    for (unsigned int j = 0; j<num_rows[1];++j)
      sparsity_pattern[rows[0][i]].insert(rows[1][j]);
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
      else
        sparsity_pattern[rows[0][i]].insert(rows[1][j]);
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
  std::vector< std::set<int> >::const_iterator set;
  for(set = sparsity_pattern.begin(); set != sparsity_pattern.end(); ++set)
    nzrow[set-sparsity_pattern.begin()] = set->size();
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
  std::vector< std::set<int> >::const_iterator set;
  for(set = sparsity_pattern.begin(); set != sparsity_pattern.end(); ++set)
    nz += set->size();
  return nz;
}
//-----------------------------------------------------------------------------
void SparsityPattern::disp() const
{ 
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
