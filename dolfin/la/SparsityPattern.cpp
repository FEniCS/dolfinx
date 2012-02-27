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

#include <dolfin/common/MPI.h>
#include <dolfin/log/log.h>
#include <dolfin/log/LogStream.h>
#include "SparsityPattern.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
SparsityPattern::SparsityPattern(uint primary_dim)
    : GenericSparsityPattern(primary_dim), distributed(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void SparsityPattern::init(const std::vector<uint>& dims,
  const std::vector<std::pair<uint, uint> >& local_range,
  const std::vector<const boost::unordered_map<uint, uint>* > off_process_owner)
{
  // Only rank 2 sparsity patterns are supported
  dolfin_assert(dims.size() == 2);

  // Check that dimensions match
  dolfin_assert(dims.size() == local_range.size());
  dolfin_assert(dims.size() == off_process_owner.size());

  // Clear sparsity pattern data
  diagonal.clear();
  off_diagonal.clear();
  non_local.clear();
  this->off_process_owner.clear();

  // Store local ownership range
  _local_range = local_range;

  // Check if sparsity pattern is distributed
  dolfin_assert(_primary_dim < local_range.size());
  dolfin_assert(local_range[_primary_dim].second > local_range[_primary_dim].first);
  const uint range0 = _local_range[_primary_dim].first;
  const uint range1 = _local_range[_primary_dim].second;
  if (range0 == 0 && MPI::max(range1) == range1)
    distributed = false;
  else
    distributed = true;

  // Store copy of nonlocal index to owning process map
  this->off_process_owner.reserve(off_process_owner.size());
  for (uint i = 0; i < off_process_owner.size(); ++i)
  {
    dolfin_assert(off_process_owner[i]);
    this->off_process_owner.push_back(*off_process_owner[i]);
  }

  // Check that primary dimension is valid
  if (_primary_dim > 1)
   {
    dolfin_error("SparsityPattern.cpp",
                 "primary dimension for sparsity pattern storage",
                 "Primary dimension must be less than 2 (0=row major, 1=column major");
  }

  // Resize diagonal block
  diagonal.resize(range1- range0);

  // Resize off-diagonal block (only needed when local range != global range)
  if (distributed)
    off_diagonal.resize(range1- range0);
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert(const std::vector<const std::vector<uint>* >& entries)
{
  dolfin_assert(entries.size() == 2);
  dolfin_assert(entries[0]);
  dolfin_assert(entries[1]);

  const std::vector<uint>* map_i;
  const std::vector<uint>* map_j;
  uint primary_codim;
  dolfin_assert(_primary_dim < 2);
  if (_primary_dim == 0)
  {
    primary_codim = 1;
    map_i = entries[0];
    map_j = entries[1];
  }
  else
  {
    primary_codim = 0;
    map_i = entries[1];
    map_j = entries[0];
  }

  // Check local range
  if (!distributed)
  {
    // Sequential mode, do simple insertion
    std::vector<uint>::const_iterator i_index;
    for (i_index = map_i->begin(); i_index != map_i->end(); ++i_index)
      diagonal[*i_index].insert(map_j->begin(), map_j->end());
  }
  else
  {
    // Parallel mode, use either diagonal, off_diagonal or non_local
    std::vector<uint>::const_iterator i_index;
    for (i_index = map_i->begin(); i_index != map_i->end(); ++i_index)
    {
      if (_local_range[_primary_dim].first <= *i_index && *i_index < _local_range[_primary_dim].second)
      {
        // Subtract offset
        const uint I = *i_index - _local_range[_primary_dim].first;

        // Store local entry in diagonal or off-diagonal block
        std::vector<uint>::const_iterator j_index;
        for (j_index = map_j->begin(); j_index != map_j->end(); ++j_index)
        {
          if (_local_range[primary_codim].first <= *j_index && *j_index < _local_range[primary_codim].second)
          {
            dolfin_assert(I < diagonal.size());
            diagonal[I].insert(*j_index);
          }
          else
          {
            dolfin_assert(I < off_diagonal.size());
            off_diagonal[I].insert(*j_index);
          }
        }
      }
      else
      {
        // Store non-local entry (communicated later during apply())
        std::vector<uint>::const_iterator j_index;
        for (j_index = map_j->begin(); j_index != map_j->end(); ++j_index)
        {
          non_local.push_back(*i_index);
          non_local.push_back(*j_index);
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
dolfin::uint SparsityPattern::rank() const
{
  return 2;
}
//-----------------------------------------------------------------------------
std::pair<dolfin::uint, dolfin::uint> SparsityPattern::local_range(uint dim) const
{
  dolfin_assert(dim < 2);
  return _local_range[dim];
}
//-----------------------------------------------------------------------------
dolfin::uint SparsityPattern::num_nonzeros() const
{
  uint nz = 0;
  typedef std::vector<set_type>::const_iterator slice_it;
  for (slice_it slice = diagonal.begin(); slice != diagonal.end(); ++slice)
    nz += slice->size();
  for (slice_it slice = off_diagonal.begin(); slice != off_diagonal.end(); ++slice)
    nz += slice->size();
  return nz;
}
//-----------------------------------------------------------------------------
void SparsityPattern::num_nonzeros_diagonal(std::vector<uint>& num_nonzeros) const
{
  // Resize vector
  num_nonzeros.resize(diagonal.size());

  // Get number of nonzeros per generalised row
  typedef std::vector<set_type>::const_iterator slice_it;
  for (slice_it slice = diagonal.begin(); slice != diagonal.end(); ++slice)
    num_nonzeros[slice - diagonal.begin()] = slice->size();
}
//-----------------------------------------------------------------------------
void SparsityPattern::num_nonzeros_off_diagonal(std::vector<uint>& num_nonzeros) const
{
  // Resize vector
  num_nonzeros.resize(off_diagonal.size());

  // Compute number of nonzeros per generalised row
  typedef std::vector<set_type>::const_iterator slice_it;
  for (slice_it slice = off_diagonal.begin(); slice != off_diagonal.end(); ++slice)
    num_nonzeros[slice - off_diagonal.begin()] = slice->size();
}
//-----------------------------------------------------------------------------
void SparsityPattern::num_local_nonzeros(std::vector<uint>& num_nonzeros) const
{
  num_nonzeros_diagonal(num_nonzeros);
  if (off_diagonal.size() > 0)
  {
    std::vector<uint> tmp;
    num_nonzeros_off_diagonal(tmp);
    dolfin_assert(num_nonzeros.size() == tmp.size());
    std::transform(num_nonzeros.begin(), num_nonzeros.end(), tmp.begin(),
                   num_nonzeros.begin(), std::plus<uint>());
  }
}
//-----------------------------------------------------------------------------
void SparsityPattern::apply()
{
  uint primary_codim;
  dolfin_assert(_primary_dim < 2);
  if (_primary_dim == 0)
    primary_codim = 1;
  else
    primary_codim = 0;

  const uint num_processes = MPI::num_processes();
  const uint proc_number = MPI::process_number();

  // Print some useful information
  if (get_log_level() <= DBG)
    info_statistics();

  // Communicate non-local blocks if any
  if (!distributed)
  {
    // Figure out correct process for each non-local entry
    dolfin_assert(non_local.size() % 2 == 0);
    std::vector<uint> destinations(non_local.size());
    for (uint i = 0; i < non_local.size(); i += 2)
    {
      // Get generalised row for non-local entry
      const uint I = non_local[i];

      // Figure out which process owns the row
      boost::unordered_map<uint, uint>::const_iterator non_local_index
          = off_process_owner[_primary_dim].find(I);
      dolfin_assert(non_local_index != off_process_owner[_primary_dim].end());
      const uint p = non_local_index->second;

      dolfin_assert(p < num_processes);
      dolfin_assert(p != proc_number);

      destinations[i] = p;
      destinations[i + 1] = p;
    }

    // Communicate non-local entries to other processes
    std::vector<uint> non_local_received;
    MPI::distribute(non_local, destinations, non_local_received);

    // Insert non-local entries received from other processes
    dolfin_assert(non_local_received.size() % 2 == 0);
    for (uint i = 0; i < non_local_received.size(); i += 2)
    {
      // Get generalised row and column
      uint I = non_local_received[i];
      const uint J = non_local_received[i + 1];

      // Sanity check
      if (I < _local_range[_primary_dim].first || I >= _local_range[_primary_dim].second)
      {
        dolfin_error("SparsityPattern.cpp",
                     "apply changes to sparsity pattern",
                     "Received illegal sparsity pattern entry for row/column %d, not in range [%d, %d]",
                     I, _local_range[_primary_dim].first, _local_range[_primary_dim].second);
      }

      // Subtract offset
      I -= _local_range[_primary_dim].first;

      // Insert in diagonal or off-diagonal block
      if (_local_range[primary_codim].first <= J && J < _local_range[primary_codim].second)
      {
        dolfin_assert(I < diagonal.size());
        diagonal[I].insert(J);
      }
      else
      {
        dolfin_assert(I < off_diagonal.size());
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
  // Print each row
  std::stringstream s;
  typedef set_type::const_iterator entry_it;
  for (uint i = 0; i < diagonal.size(); i++)
  {
    if (_primary_dim == 0)
      s << "Row " << i << ":";
    else
      s << "Col " << i << ":";

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
  const uint size0 = MPI::max(_local_range[0].second);
  const uint size1 = MPI::max(_local_range[1].second);
  cout << "Matrix of size " << size0 << " x " << size1<< " has "
       << num_nonzeros_total << " (" << 100.0*num_nonzeros_total/(size0*size1)
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
