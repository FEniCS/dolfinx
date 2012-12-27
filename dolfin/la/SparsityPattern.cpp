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
SparsityPattern::SparsityPattern(std::size_t primary_dim)
    : GenericSparsityPattern(primary_dim), distributed(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
SparsityPattern::SparsityPattern(const std::vector<std::size_t>& dims,
  const std::vector<std::pair<std::size_t, std::size_t> >& local_range,
  const std::vector<const boost::unordered_map<std::size_t, std::size_t>* > off_process_owner,
    std::size_t primary_dim)
  : GenericSparsityPattern(primary_dim), distributed(false)
{
  init(dims, local_range, off_process_owner);
}
//-----------------------------------------------------------------------------
void SparsityPattern::init(const std::vector<std::size_t>& dims,
  const std::vector<std::pair<std::size_t, std::size_t> >& local_range,
  const std::vector<const boost::unordered_map<std::size_t, std::size_t>* > off_process_owner)
{
  // Only rank 2 sparsity patterns are supported
  dolfin_assert(dims.size() == 2);

  const std::size_t _primary_dim = primary_dim();

  // Check that dimensions match
  dolfin_assert(dims.size() == local_range.size());
  dolfin_assert(dims.size() == off_process_owner.size());

  // Clear sparsity pattern data
  diagonal.clear();
  off_diagonal.clear();
  non_local.clear();
  this->off_process_owner.clear();

  // Set ownership range
  _local_range = local_range;

  // Store copy of nonlocal index to owning process map
  this->off_process_owner.reserve(off_process_owner.size());
  for (std::size_t i = 0; i < off_process_owner.size(); ++i)
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

  // Check if sparsity pattern is distributed
  if (_local_range[_primary_dim].first != 0 || _local_range[_primary_dim].second != dims[_primary_dim])
    distributed = true;

  // Resize diagonal block
  dolfin_assert(_local_range[_primary_dim].second > _local_range[_primary_dim].first);
  diagonal.resize(_local_range[_primary_dim].second - _local_range[_primary_dim].first);

  // Resize off-diagonal block (only needed when local range != global range)
  if (distributed)
    off_diagonal.resize(_local_range[_primary_dim].second - _local_range[_primary_dim].first);
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert(const std::vector<const std::vector<DolfinIndex>* >& entries)
{
  dolfin_assert(entries.size() == 2);
  dolfin_assert(entries[0]);
  dolfin_assert(entries[1]);

  const std::size_t _primary_dim = primary_dim();

  const std::vector<DolfinIndex>* map_i;
  const std::vector<DolfinIndex>* map_j;
  std::size_t primary_codim;
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

  const std::pair<DolfinIndex, DolfinIndex> local_range0(_local_range[_primary_dim].first,
                                                         _local_range[_primary_dim].second);
  const std::pair<DolfinIndex, DolfinIndex> local_range1(_local_range[primary_codim].first,
                                                         _local_range[primary_codim].second);

  // Check local range
  if (!distributed)
  {
    // Sequential mode, do simple insertion
    std::vector<DolfinIndex>::const_iterator i_index;
    for (i_index = map_i->begin(); i_index != map_i->end(); ++i_index)
      diagonal[*i_index].insert(map_j->begin(), map_j->end());
  }
  else
  {
    // Parallel mode, use either diagonal, off_diagonal or non_local
    std::vector<DolfinIndex>::const_iterator i_index;
    for (i_index = map_i->begin(); i_index != map_i->end(); ++i_index)
    {
      if (local_range0.first <= *i_index && *i_index < local_range0.second)
      {
        // Subtract offset
        const std::size_t I = *i_index - local_range0.first;

        // Store local entry in diagonal or off-diagonal block
        std::vector<DolfinIndex>::const_iterator j_index;
        for (j_index = map_j->begin(); j_index != map_j->end(); ++j_index)
        {
          if (local_range1.first <= *j_index && *j_index < local_range1.second)
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
        std::vector<DolfinIndex>::const_iterator j_index;
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
void SparsityPattern::add_edges(const std::pair<DolfinIndex, std::size_t>& vertex,
                                const std::vector<DolfinIndex>& edges)
{
  const std::size_t _primary_dim = primary_dim();
  const DolfinIndex vertex_index = vertex.first;


  std::pair<DolfinIndex, DolfinIndex> local_range(_local_range[_primary_dim].first,
                                                  _local_range[_primary_dim].second);

  // Add off-process owner if vertex is not owned by this process
  if (vertex_index < local_range.first || vertex_index >= local_range.second)
    off_process_owner[_primary_dim].insert(vertex);

  // Add edges
  std::vector<DolfinIndex> dofs0(1, vertex.first);
  std::vector<const std::vector<DolfinIndex>* > entries(2);
  entries[0] = &dofs0;
  entries[1] = &edges;
  insert(entries);
}
//-----------------------------------------------------------------------------
std::size_t SparsityPattern::rank() const
{
  return 2;
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t> SparsityPattern::local_range(std::size_t dim) const
{
  dolfin_assert(dim < 2);
  return _local_range[dim];
}
//-----------------------------------------------------------------------------
std::size_t SparsityPattern::num_nonzeros() const
{
  std::size_t nz = 0;
  typedef std::vector<set_type>::const_iterator slice_it;
  for (slice_it slice = diagonal.begin(); slice != diagonal.end(); ++slice)
    nz += slice->size();
  for (slice_it slice = off_diagonal.begin(); slice != off_diagonal.end(); ++slice)
    nz += slice->size();
  return nz;
}
//-----------------------------------------------------------------------------
void SparsityPattern::num_nonzeros_diagonal(std::vector<std::size_t>& num_nonzeros) const
{
  // Resize vector
  num_nonzeros.resize(diagonal.size());

  // Get number of nonzeros per generalised row
  typedef std::vector<set_type>::const_iterator slice_it;
  for (slice_it slice = diagonal.begin(); slice != diagonal.end(); ++slice)
    num_nonzeros[slice - diagonal.begin()] = slice->size();
}
//-----------------------------------------------------------------------------
void SparsityPattern::num_nonzeros_off_diagonal(std::vector<std::size_t>& num_nonzeros) const
{
  // Resize vector
  num_nonzeros.resize(off_diagonal.size());

  // Compute number of nonzeros per generalised row
  typedef std::vector<set_type>::const_iterator slice_it;
  for (slice_it slice = off_diagonal.begin(); slice != off_diagonal.end(); ++slice)
    num_nonzeros[slice - off_diagonal.begin()] = slice->size();
}
//-----------------------------------------------------------------------------
void SparsityPattern::num_local_nonzeros(std::vector<std::size_t>& num_nonzeros) const
{
  num_nonzeros_diagonal(num_nonzeros);
  if (!off_diagonal.empty())
  {
    std::vector<std::size_t> tmp;
    num_nonzeros_off_diagonal(tmp);
    dolfin_assert(num_nonzeros.size() == tmp.size());
    std::transform(num_nonzeros.begin(), num_nonzeros.end(), tmp.begin(),
                   num_nonzeros.begin(), std::plus<std::size_t>());
  }
}
//-----------------------------------------------------------------------------
void SparsityPattern::get_edges(std::size_t vertex, std::vector<DolfinIndex>& edges) const
{
  dolfin_assert(vertex >= _local_range[0].first && vertex < _local_range[0].second);

  const std::size_t local_vertex = vertex - _local_range[0].first;
  dolfin_assert(local_vertex < diagonal.size());
  std::size_t size = diagonal[local_vertex].size();
  if (!off_diagonal.empty())
  {
    dolfin_assert(local_vertex < off_diagonal.size());
    size += off_diagonal[local_vertex].size();
  }
  edges.resize(size);

  std::copy(diagonal[local_vertex].begin(), diagonal[local_vertex].end(), edges.begin());
  if (!off_diagonal.empty())
  {
    std::copy(off_diagonal[local_vertex].begin(),
              off_diagonal[local_vertex].end(),
              edges.begin() + diagonal[local_vertex].size());
  }
}
//-----------------------------------------------------------------------------
void SparsityPattern::apply()
{
  const std::size_t _primary_dim = primary_dim();

  std::size_t primary_codim;
  dolfin_assert(_primary_dim < 2);
  if (_primary_dim == 0)
    primary_codim = 1;
  else
    primary_codim = 0;

  const std::size_t num_processes = MPI::num_processes();
  const std::size_t proc_number = MPI::process_number();

  // Print some useful information
  if (get_log_level() <= DBG)
    info_statistics();

  // Communicate non-local blocks if any
  if (distributed)
  {
    // Figure out correct process for each non-local entry
    dolfin_assert(non_local.size() % 2 == 0);
    std::vector<std::size_t> destinations(non_local.size());
    for (std::size_t i = 0; i < non_local.size(); i += 2)
    {
      // Get generalised row for non-local entry
      const std::size_t I = non_local[i];
      cout << "Trying to find row I = " << I << endl;

      // Figure out which process owns the row
      boost::unordered_map<std::size_t, std::size_t>::const_iterator non_local_index
          = off_process_owner[_primary_dim].find(I);
      dolfin_assert(non_local_index != off_process_owner[_primary_dim].end());
      const std::size_t p = non_local_index->second;

      dolfin_assert(p < num_processes);
      dolfin_assert(p != proc_number);

      destinations[i] = p;
      destinations[i + 1] = p;
    }

    // Communicate non-local entries to other processes
    std::vector<std::size_t> non_local_received;
    MPI::distribute(non_local, destinations, non_local_received);

    // Insert non-local entries received from other processes
    dolfin_assert(non_local_received.size() % 2 == 0);
    for (std::size_t i = 0; i < non_local_received.size(); i += 2)
    {
      // Get generalised row and column
      std::size_t I = non_local_received[i];
      const std::size_t J = non_local_received[i + 1];

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
std::string SparsityPattern::str(bool verbose) const
{
  // Print each row
  std::stringstream s;
  typedef set_type::const_iterator entry_it;
  for (std::size_t i = 0; i < diagonal.size(); i++)
  {
    if (primary_dim() == 0)
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
std::vector<std::vector<std::size_t> > SparsityPattern::diagonal_pattern(Type type) const
{
  std::vector<std::vector<std::size_t> > v(diagonal.size());
  for (std::size_t i = 0; i < diagonal.size(); ++i)
    v[i].insert(v[i].begin(), diagonal[i].begin(), diagonal[i].end());

  if (type == sorted)
  {
    for (std::size_t i = 0; i < v.size(); ++i)
      std::sort(v[i].begin(), v[i].end());
  }

  return v;
}
//-----------------------------------------------------------------------------
std::vector<std::vector<std::size_t> > SparsityPattern::off_diagonal_pattern(Type type) const
{
  std::vector<std::vector<std::size_t> > v(off_diagonal.size());
  for (std::size_t i = 0; i < off_diagonal.size(); ++i)
    v[i].insert(v[i].begin(), off_diagonal[i].begin(), off_diagonal[i].end());

  if (type == sorted)
  {
    for (std::size_t i = 0; i < v.size(); ++i)
      std::sort(v[i].begin(), v[i].end());
  }

  return v;
}
//-----------------------------------------------------------------------------
void SparsityPattern::info_statistics() const
{
  // Count nonzeros in diagonal block
  std::size_t num_nonzeros_diagonal = 0;
  for (std::size_t i = 0; i < diagonal.size(); ++i)
    num_nonzeros_diagonal += diagonal[i].size();

  // Count nonzeros in off-diagonal block
  std::size_t num_nonzeros_off_diagonal = 0;
  for (std::size_t i = 0; i < off_diagonal.size(); ++i)
    num_nonzeros_off_diagonal += off_diagonal[i].size();

  // Count nonzeros in non-local block
  const std::size_t num_nonzeros_non_local = non_local.size() / 2;

  // Count total number of nonzeros
  const std::size_t num_nonzeros_total
    = num_nonzeros_diagonal + num_nonzeros_off_diagonal + num_nonzeros_non_local;

  std::size_t size0 = _local_range[0].second;
  std::size_t size1 = _local_range[1].second;
  if (distributed)
  {
    size0 = MPI::max(size0);
    size1 = MPI::max(size1);
  }

  // Return number of entries
  cout << "Matrix of size " << size0 << " x " << size1 << " has "
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
