// Copyright (C) 2007-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Magnus Vikstrom, 2008.
// Modified by Anders Logg, 2008-2009.
//
// First added:  2007-03-13
// Last changed: 2009-05-23

#include <algorithm>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/main/MPI.h>
#include "SparsityPattern.h"

using namespace dolfin;

// Typedef of iterators for convenience
typedef std::vector<std::vector<dolfin::uint> >::iterator iterator;
typedef std::vector<std::vector<dolfin::uint> >::const_iterator const_iterator;

//-----------------------------------------------------------------------------
SparsityPattern::SparsityPattern() : parallel(MPI::num_processes() > 1)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
SparsityPattern::~SparsityPattern()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void SparsityPattern::init(uint rank, const uint* dims)
{
  // Check rank
  if (rank < 1 or rank > 2)
    error("Sparsity pattern only implemented for tensors of rank 1 and 2.");

  // Store dimensions
  shape.resize(rank);
  for (uint i = 0; i < rank; ++i)
    shape[i] = dims[i];

  // Clear sparsity pattern
  diagonal.clear();
  off_diagonal.clear();

  // FIXME: This is wrong when running in parallel
  diagonal.resize(shape[0]);
  off_diagonal.resize(shape[0]);
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert(const uint* num_rows, const uint * const * rows)
{
  const uint  m = num_rows[0];
  const uint  n = num_rows[1];
  const uint* r = rows[0];
  const uint* c = rows[1];

  if (parallel)
  {
    error("Implement me!");
  }

  for (uint i = 0; i < m; ++i)
  {
    for (uint j = 0; j < n; ++j)
    {
      bool inserted = false;
      uint k = 0;
      while (k < diagonal[r[i]].size() && !inserted)
      {
        if (c[j] == diagonal[r[i]][k])
          inserted = true;
        ++k;
      }
      if (!inserted)
        diagonal[r[i]].push_back(c[j]);
    }
  }
}
//-----------------------------------------------------------------------------
void SparsityPattern::sort()
{
  for (iterator it = diagonal.begin(); it != diagonal.end(); ++it)
    std::sort(it->begin(), it->end()); 
}
//-----------------------------------------------------------------------------
dolfin::uint SparsityPattern::size(uint i) const
{
  dolfin_assert(i < shape.size());
  return shape[i];
}
//-----------------------------------------------------------------------------
dolfin::uint SparsityPattern::num_nonzeros() const
{
  uint nz = 0;
  for (const_iterator it = diagonal.begin(); it != diagonal.end(); ++it)
    nz += it->size();
  return nz;
}
//-----------------------------------------------------------------------------
void SparsityPattern::num_nonzeros_diagonal(uint* num_nonzeros) const
{
  // Check rank
  if (shape.size() != 2)
    error("Non-zero entries per row can be computed for matrices only.");

  // Compute number of nonzeros per row
  std::vector< std::vector<uint> >::const_iterator row;
  for (row = diagonal.begin(); row != diagonal.end(); ++row)
    num_nonzeros[row - diagonal.begin()] = row->size();
}
//-----------------------------------------------------------------------------
void SparsityPattern::num_nonzeros_off_diagonal(uint* num_nonzeros) const
{
  /*
  if ( dim[1] == 0 )
    error("Non-zero entries per row can be computed for matrices only.");

  if ( diagonal.size() == 0 )
    error("Sparsity pattern has not been computed.");

  // Compute number of nonzeros per row diagonal and off-diagonal
  uint offset = range[process_number];
  for(uint i = 0; i+offset<range[process_number+1]; ++i)
  {
    d_nzrow[i] = diagonal[i+offset].size();
    o_nzrow[i] = o_diagonal[i+offset].size();
  }
  */
}
//-----------------------------------------------------------------------------
void SparsityPattern::apply()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::string SparsityPattern::str() const
{
  // Check rank
  if (shape.size() != 2)
    error("Sparsity pattern can only be displayed for matrices.");

  // Print each row
  std::stringstream s;
  for (uint i = 0; i < diagonal.size(); i++)
  {
    s << "Row " << i << ":";
    for (uint k = 0; k < diagonal[i].size(); ++k)
      cout << " " << diagonal[i][k];
    s << std::endl;
  }

  return s.str();
}
//-----------------------------------------------------------------------------
const std::vector<std::vector<dolfin::uint> >& SparsityPattern::pattern() const
{
  return diagonal;
}
//-----------------------------------------------------------------------------

/*
void SparsityPattern::process_range(uint process_number, uint local_range[])
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
void SparsityPattern::init_range()
{
  uint num_procs = dolfin::MPI::num_processes();
  range = new uint[num_procs+1];
  range[0] = 0;

  for(uint p=0; p<num_procs; ++p)
  {
    range[p+1] = range[p] + dim[0]/num_procs + ((dim[0]%num_procs) > p ? 1 : 0);
    info("range[%d] = %d", p+1, range[p+1]);
  }
}
//-----------------------------------------------------------------------------
*/
