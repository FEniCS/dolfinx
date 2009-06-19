// Copyright (C) 2007-2009 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Magnus Vikstrom, 2008.
// Modified by Anders Logg, 2008-2009.
//
// First added:  2007-03-13
// Last changed: 2009-06-19

#include <algorithm>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/main/MPI.h>
#include "SparsityPattern.h"

using namespace dolfin;

// Typedef of iterators for convenience
typedef std::vector<std::vector<dolfin::uint> >::iterator iterator;
typedef std::vector<std::vector<dolfin::uint> >::const_iterator const_iterator;

// Inlined function for insertion
inline void insert_entry(unsigned int j, std::vector<unsigned int>& row)
{
  if (std::find(row.begin(), row.end(), j) == row.end())
    row.push_back(j);
}

//-----------------------------------------------------------------------------
SparsityPattern::SparsityPattern(Type type) 
  : type(type), _sorted(false), rmin(0), rmax(0)
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
  // Store dimensions
  shape.resize(rank);
  for (uint i = 0; i < rank; ++i)
    shape[i] = dims[i];

  // Clear sparsity pattern
  diagonal.clear();
  off_diagonal.clear();
  non_local.clear();

  // Check rank, ignore if not a matrix
  if (shape.size() != 2)
    return;

  // Get local range
  std::pair<uint, uint> r = range();
  rmin = r.first;
  rmax = r.second;

  // Resize diagonal block
  diagonal.resize(rmax - rmin);

  // Resize off-diagonal block (only needed when local range != global range)
  if (rmin != 0 || rmax != shape[0])
    off_diagonal.resize(rmax - rmin);
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert(const uint* num_rows, const uint * const * rows)
{
  // Check rank, ignore if not a matrix
  if (shape.size() != 2)
    return;

  // Set sorted flag to false
  _sorted = false;

  // Get local rows and columsn to insert
  const uint  m = num_rows[0];
  const uint  n = num_rows[1];
  const uint* r = rows[0];
  const uint* c = rows[1];

  // Check local range
  if (rmin == 0 && rmax == shape[0])
  {
    // Sequential mode, do simple insertion
    for (uint i = 0; i < m; ++i)
    {
      std::vector<uint>& row = diagonal[r[i]];
      for (uint j = 0; j < n; ++j)
      {
        insert_entry(c[j], row);
      }
    }
  }

  /*
  else
  {
    // Parallel mode, use either diagonal, off_diagonal or non_local

    std::pair<uint, uint> _range = range();
    const uint ind_start = _range.first;
    const uint ind_stop = _range.second;

    for (uint i = 0; i < m; ++i)
    {
      if (r[i] >= ind_start and r[i] < ind_stop)
      {
        for (uint j = 0; j < n; ++j)
        {
          if (c[j] >= ind_start and r[i] < ind_stop)
          {
            // Insert entry on diagonal if not already inserted
            if (std::find(diagonal[r[i]].begin(), diagonal[r[i]].end(), c[j]) == diagonal[r[i]].end())
              diagonal[r[i]].push_back(c[j]);
          }
          else
          {
            // Insert entry on off-diagonal if not already inserted
            if (std::find(off_diagonal[r[i]].begin(), off_diagonal[r[i]].end(), c[j]) == off_diagonal[r[i]].end())
              off_diagonal[r[i]].push_back(c[j]);
          }
        }
      }
      else
      {
        // FIXME: Unhandled case. We need to store these indexes and send them to the corresponding processor
      }
    }
  }
    */

}
//-----------------------------------------------------------------------------
dolfin::uint SparsityPattern::rank() const
{
  return shape.size();
}
//-----------------------------------------------------------------------------
dolfin::uint SparsityPattern::size(uint i) const
{
  dolfin_assert(i < shape.size());
  return shape[i];
}
//-----------------------------------------------------------------------------
std::pair<dolfin::uint, dolfin::uint> SparsityPattern::range() const
{
  return MPI::local_range(size(0));
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
  // Sort sparsity pattern if required
  if (type == sorted && _sorted == false)
  {
    cout << "Sorting pattern " << endl;
    sort();
    _sorted = true;
  }
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
  if (type == sorted && _sorted == false)
    error("SparsityPattern has not been sorted. You need to call SparsityPattern::apply().");

  return diagonal;
}
//-----------------------------------------------------------------------------
void SparsityPattern::sort()
{
  for (iterator it = diagonal.begin(); it != diagonal.end(); ++it)
    std::sort(it->begin(), it->end()); 
}
//-----------------------------------------------------------------------------
