// Copyright (C) 2007-2011 Garth N. Wells
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
// Modified by Magnus Vikstrom, 2008.
// Modified by Anders Logg, 2008-2009.
// Modified by Ola Skavhaug, 2009.
//
// First added:  2007-03-13
// Last changed: 2011-01-02

#include <algorithm>

#include <dolfin/log/log.h>
#include <dolfin/log/LogStream.h>
#include <dolfin/common/MPI.h>
#include "SparsityPattern.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
SparsityPattern::SparsityPattern() : row_range_min(0), row_range_max(0),
                                     col_range_min(0), col_range_max(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void SparsityPattern::init(const std::vector<uint>& dims,
  const std::vector<std::pair<uint, uint> >& ownership_range,
  const std::vector<const boost::unordered_map<uint, uint>* > off_process_owner)
{
  // Only rank 1 and 2 sparsity patterns are supported
  assert(dims.size() < 3);

  // Check that dimensions match
  assert(dims.size() == ownership_range.size());
  assert(dims.size() == off_process_owner.size());

  // Store dimensions
  shape = dims;

  // Set ownership range
  this->ownership_range = ownership_range;

  // Store copy of nonlocal index to owning process map
  for (uint i = 0; i < off_process_owner.size(); ++i)
  {
    assert(off_process_owner[i]);
    this->off_process_owner.push_back(*off_process_owner[i]);
  }

  // Clear sparsity pattern data
  diagonal.clear();
  off_diagonal.clear();
  non_local.clear();

  // Check rank, ignore if not a matrix
  if (shape.size() != 2)
    return;

  // Set ownership range
  this->ownership_range = ownership_range;

  // Get local range
  row_range_min = this->ownership_range[0].first;
  row_range_max = this->ownership_range[0].second;
  col_range_min = this->ownership_range[1].first;
  col_range_max = this->ownership_range[1].second;

  // Resize diagonal block
  assert(row_range_max > row_range_min);
  diagonal.resize(row_range_max - row_range_min);

  // Resize off-diagonal block (only needed when local range != global range)
  if (row_range_min != 0 || row_range_max != shape[0])
    off_diagonal.resize(row_range_max - row_range_min);
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert(const std::vector<const std::vector<uint>* >& entries)
{
  // Check rank, ignore if not a matrix
  if (shape.size() != 2)
    return;

  assert(entries.size() == 2);
  assert(entries[0]);
  assert(entries[1]);

  // Get local rows and columns to insert
  const std::vector<uint>& map_i = *entries[0];
  const std::vector<uint>& map_j = *entries[1];

  // Check local range
  if (row_range_min == 0 && row_range_max == shape[0])
  {
    // Sequential mode, do simple insertion
    for (std::vector<uint>::const_iterator row = map_i.begin(); row != map_i.end(); ++row)
      diagonal[*row].insert(map_j.begin(), map_j.end());
  }
  else
  {
    // Parallel mode, use either diagonal, off_diagonal or non_local
    for (std::vector<uint>::const_iterator row = map_i.begin(); row != map_i.end(); ++row)
    {
      if (row_range_min <= *row && *row < row_range_max)
      {
        // Subtract offset
        const uint I = *row - row_range_min;

        // Store local entry in diagonal or off-diagonal block
        for (std::vector<uint>::const_iterator col = map_j.begin(); col != map_j.end(); ++col)
        {
          if (col_range_min <= *col && *col < col_range_max)
          {
            assert(I < diagonal.size());
            diagonal[I].insert(*col);
          }
          else
          {
            assert(I < off_diagonal.size());
            off_diagonal[I].insert(*col);
          }
        }
      }
      else
      {
        // Store non-local entry (communicated later during apply())
        for (std::vector<uint>::const_iterator col = map_j.begin(); col != map_j.end(); ++col)
        {
          non_local.push_back(*row);
          non_local.push_back(*col);
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
  assert(i < shape.size());
  return shape[i];
}
//-----------------------------------------------------------------------------
std::pair<dolfin::uint, dolfin::uint> SparsityPattern::local_range(uint dim) const
{
  assert(dim < 2);
  return ownership_range[dim];
}
//-----------------------------------------------------------------------------
dolfin::uint SparsityPattern::num_nonzeros() const
{
  uint nz = 0;
  typedef std::vector<set_type>::const_iterator row_it;
  for (row_it row = diagonal.begin(); row != diagonal.end(); ++row)
    nz += row->size();
  for (row_it row = off_diagonal.begin(); row != off_diagonal.end(); ++row)
    nz += row->size();
  return nz;
}
//-----------------------------------------------------------------------------
void SparsityPattern::num_nonzeros_diagonal(std::vector<uint>& num_nonzeros) const
{
  // Check rank
  if (shape.size() != 2)
  {
    dolfin_error("SparsityPattern.cpp",
                 "access number of nonzero diagonal entries",
                 "Non-zero entries per row can be computed for matrices only");
  }

  // Resize vector
  num_nonzeros.resize(diagonal.size());

  // Get number of nonzeros per row
  typedef std::vector<set_type>::const_iterator row_it;
  for (row_it row = diagonal.begin(); row != diagonal.end(); ++row)
    num_nonzeros[row - diagonal.begin()] = row->size();
}
//-----------------------------------------------------------------------------
void SparsityPattern::num_nonzeros_off_diagonal(std::vector<uint>& num_nonzeros) const
{
  // Check rank
  if (shape.size() != 2)
  {
    dolfin_error("SparsityPattern.cpp",
                 "access number of nonzero off-diagonal entries",
                 "Non-zero entries per row can be computed for matrices only");
  }

  // Resize vector
  num_nonzeros.resize(off_diagonal.size());

  // Compute number of nonzeros per row
  typedef std::vector<set_type>::const_iterator row_it;
  for (row_it row = off_diagonal.begin(); row != off_diagonal.end(); ++row)
    num_nonzeros[row - off_diagonal.begin()] = row->size();
}
//-----------------------------------------------------------------------------
void SparsityPattern::apply()
{
  // Check rank, ignore if not a matrix
  if (shape.size() != 2)
    return;

  const uint num_processes = MPI::num_processes();
  const uint proc_number = MPI::process_number();

  // Print some useful information
  if (get_log_level() <= DBG)
    info_statistics();

  // Communicate non-local blocks if any
  if (row_range_min != 0 || row_range_max != shape[0])
  {
    // Figure out correct process for each non-local entry
    assert(non_local.size() % 2 == 0);
    std::vector<uint> destinations(non_local.size());
    for (uint i = 0; i < non_local.size(); i += 2)
    {
      // Get row for non-local entry
      const uint I = non_local[i];

      // Figure out which process owns the row
      boost::unordered_map<uint, uint>::const_iterator non_local_index
          = off_process_owner[0].find(I);
      assert(non_local_index != off_process_owner[0].end());
      const uint p = non_local_index->second;

      assert(p < num_processes);
      assert(p != proc_number);

      destinations[i] = p;
      destinations[i + 1] = p;
    }

    // Communicate non-local entries to other processes
    std::vector<uint> non_local_received;
    MPI::distribute(non_local, destinations, non_local_received);

    // Insert non-local entries received from other processes
    assert(non_local_received.size() % 2 == 0);
    for (uint i = 0; i < non_local_received.size(); i+= 2)
    {
      // Get row and column
      uint I = non_local_received[i];
      const uint J = non_local_received[i + 1];

      // Sanity check
      if (I < row_range_min || I >= row_range_max)
      {
        dolfin_error("SparsityPattern.cpp",
                     "apply changes to sparsity pattern",
                     "Received illegal sparsity pattern entry for row %d, not in range [%d, %d]",
                     I, row_range_min, row_range_max);
      }

      // Subtract offset
      I -= row_range_min;

      // Insert in diagonal or off-diagonal block
      if (col_range_min <= J && J < col_range_max)
      {
        assert(I < diagonal.size());
        diagonal[I].insert(J);
      }
      else
      {
        assert(I < off_diagonal.size());
        off_diagonal[I].insert(J);
      }
    }
  }

  // Clear non-local entries
  non_local.clear();
}
//-----------------------------------------------------------------------------
std::string SparsityPattern::str() const
{
  // Check rank
  if (shape.size() != 2)
  {
    dolfin_error("SparsityPattern.cpp",
                 "return string representation of sparsity pattern",
                 "Only available for matrices");
  }

  // Print each row
  std::stringstream s;
  typedef set_type::const_iterator entry_it;
  for (uint i = 0; i < diagonal.size(); i++)
  {
    s << "Row " << i << ":";
    for (entry_it entry = diagonal[i].begin(); entry != diagonal[i].end(); ++entry)
      s << " " << *entry;
    s << std::endl;
  }

  return s.str();
}
//-----------------------------------------------------------------------------
std::vector<std::vector<dolfin::uint> > SparsityPattern::diagonal_pattern(Type type) const
{
  std::vector<std::vector<uint> > v(diagonal.size());
  for (uint i = 0; i < diagonal.size(); ++i)
    v[i].insert(v[i].begin(), diagonal[i].begin(), diagonal[i].end());

  if (type == sorted)
  {
    for (uint i = 0; i < v.size(); ++i)
      std::sort(v[i].begin(), v[i].end());
  }

  return v;
}
//-----------------------------------------------------------------------------
std::vector<std::vector<dolfin::uint> > SparsityPattern::off_diagonal_pattern(Type type) const
{
  std::vector<std::vector<uint> > v(off_diagonal.size());
  for (uint i = 0; i < off_diagonal.size(); ++i)
    v[i].insert(v[i].begin(), off_diagonal[i].begin(), off_diagonal[i].end());

  if (type == sorted)
  {
    for (uint i = 0; i < v.size(); ++i)
      std::sort(v[i].begin(), v[i].end());
  }

  return v;
}
//-----------------------------------------------------------------------------
void SparsityPattern::info_statistics() const
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
  const uint num_nonzeros_total
    = num_nonzeros_diagonal + num_nonzeros_off_diagonal + num_nonzeros_non_local;

  // Return number of entries
  cout << "Matrix of size " << shape[0] << " x " << shape[1] << " has "
       << num_nonzeros_total << " (" << 100.0*num_nonzeros_total/(shape[0]*shape[1])
        << "%)" << " nonzero entries." << endl;
  if (num_nonzeros_total != num_nonzeros_diagonal)
  {
    cout << "Diagonal: " << num_nonzeros_diagonal << " ("
         << (100.0*static_cast<double>(num_nonzeros_diagonal) / static_cast<double>(num_nonzeros_total))
         << "%), ";
    cout << "off-diagonal: " << num_nonzeros_off_diagonal << " ("
         << (100.0*static_cast<double>(num_nonzeros_off_diagonal)/static_cast<double>(num_nonzeros_total))
         << "%), ";
    cout << "non-local: " << num_nonzeros_non_local << " ("
         << (100.0*static_cast<double>(num_nonzeros_non_local)/static_cast<double>(num_nonzeros_total))
         << "%)";
    cout << endl;
  }
}
//-----------------------------------------------------------------------------
