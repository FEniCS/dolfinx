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
inline void insert_column(unsigned int j, std::vector<unsigned int>& columns)
{
  if (std::find(columns.begin(), columns.end(), j) == columns.end())
    columns.push_back(j);
}

//-----------------------------------------------------------------------------
SparsityPattern::SparsityPattern(Type type) 
  : type(type), _sorted(false), range_min(0), range_max(0)
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
  range_min = r.first;
  range_max = r.second;

  // Resize diagonal block
  diagonal.resize(range_max - range_min);

  // Resize off-diagonal block (only needed when local range != global range)
  if (range_min != 0 || range_max != shape[0])
    off_diagonal.resize(range_max - range_min);
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
  const uint* map_i = rows[0];
  const uint* map_j = rows[1];

  // Check local range
  if (range_min == 0 && range_max == shape[0])
  {
    // Sequential mode, do simple insertion
    for (uint i = 0; i < m; ++i)
    {
      const uint I = map_i[i];
      for (uint j = 0; j < n; ++j)
      {
        insert_column(map_j[j], diagonal[I]);
      }
    }
  }
  else
  {
    // Parallel mode, use either diagonal, off_diagonal or non_local
    for (uint i = 0; i < m; ++i)
    {
      const uint I = map_i[i];
      if (range_min <= I && I < range_max)
      {
        // Store local entry in diagonal or off-diagonal block
        for (uint j = 0; j < n; ++j)
        {
          const uint J = map_j[j];
          if (range_min <= J && J < range_max)
            insert_column(J, diagonal[I]);
          else
            insert_column(J, off_diagonal[I]);
        }
      }
      else
      {
        // Store non-local entry and communicate during apply()
        for (uint j = 0; j < n; ++j)
        {
          const uint J = map_j[j];
          non_local.push_back(I);
          non_local.push_back(J);
        }
      }
    }
  }
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
  dolfin_debug("Calling apply() for sparsity pattern.");

  // Write some statistics
  info(statistics());

  // Communicate non-local blocks if any
  if (range_min != 0 || range_max != shape[0])
  {
    // Figure out correct process for each non-local entry
    dolfin_assert(non_local.size() % 2 == 0);
    std::vector<uint> partition(non_local.size());
    const uint process_number = MPI::process_number();
    for (uint i = 0; i < non_local.size(); i+= 2)
    {
      // Get row for non-local entry
      const uint I = non_local[i];

      // Figure out which process owns the row
      const uint p = MPI::index_owner(I, shape[0]);
      dolfin_assert(p < MPI::num_processes());
      dolfin_assert(p != process_number);
      partition[i] = p;
      partition[i + 1] = p;
    }

    info("Communicating %d non-local sparsity pattern entries.", non_local.size() / 2);

    // Communicate non-local entries
    MPI::distribute(non_local, partition);

    info("Received %d non-local sparsity pattern entries.", non_local.size() / 2);

    // Insert non-local entries received from other processes
    dolfin_assert(non_local.size() % 2 == 0);
    for (uint i = 0; i < non_local.size(); i+= 2)
    {
      // Get row and column
      const uint I = non_local[i];
      const uint J = non_local[i + 1];

      // Sanity check
      if (I < range_min || I >= range_max)
        error("Received illegal sparsity pattern entry for row %d, not in range [%d, %d].",
              I, range_min, range_max);
      
      // Insert in diagonal or off-diagonal block
      if (range_min <= J && J < range_max)
        insert_column(J, diagonal[I]);
      else
        insert_column(J, off_diagonal[I]);
    }

    // Clear non-local entries
    non_local.clear();
  }

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
std::string SparsityPattern::statistics() const
{
   // Count nonzeros in diagonal block
  uint num_nonzeros_diagonal = 0;
  for (uint i = 0; i < diagonal.size(); ++i)
    num_nonzeros_diagonal += diagonal[i].size();

  // Count nonzeros in off-diagonal block
  uint num_nonzeros_off_diagonal = 0;
  for (uint i = 0; i < off_diagonal.size(); ++i)
    num_nonzeros_off_diagonal += off_diagonal[i].size();

  // Count nonzeros in non-local block
  const uint num_nonzeros_non_local = non_local.size() / 2;

  // Count total number of nonzeros
  const uint num_nonzeros_total = 
    num_nonzeros_diagonal + num_nonzeros_off_diagonal + num_nonzeros_non_local;

  // Return number of entries
  std::stringstream s;
  s << "Nonzeros in sparsity pattern: ";
  s << "diagonal = " << num_nonzeros_diagonal << " ("
    << (100.0 * static_cast<double>(num_nonzeros_diagonal) / static_cast<double>(num_nonzeros_total))
    << "\%), ";
  s << "off-diagonal = " << num_nonzeros_off_diagonal << " ("
    << (100.0 * static_cast<double>(num_nonzeros_off_diagonal) / static_cast<double>(num_nonzeros_total))
    << "\%), ";
  s << "non-local = " << num_nonzeros_non_local << " ("
    << (100.0 * static_cast<double>(num_nonzeros_non_local) / static_cast<double>(num_nonzeros_total))
    << "\%), ";
  s << "total = " << num_nonzeros_total;

  return s.str();
}
//-----------------------------------------------------------------------------
