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
// Last changed: 2014-11-26

#include <algorithm>

#include <dolfin/common/MPI.h>
#include <dolfin/log/LogStream.h>
#include <dolfin/la/IndexMap.h>
#include "SparsityPattern.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
SparsityPattern::SparsityPattern(MPI_Comm comm, std::size_t primary_dim)
  : _primary_dim(primary_dim), _mpi_comm(comm)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
SparsityPattern::SparsityPattern(MPI_Comm comm,
  const std::vector<std::shared_ptr<const IndexMap>> index_maps,
  std::size_t primary_dim)
  : _primary_dim(primary_dim), _mpi_comm(comm)
{
  init(index_maps);
}
//-----------------------------------------------------------------------------
void SparsityPattern::init(const std::vector<std::shared_ptr<const IndexMap>> index_maps)
{
  // Only rank 2 sparsity patterns are supported
  dolfin_assert(index_maps.size() == 2);

  _index_maps = index_maps;

  const std::size_t _primary_dim = primary_dim();

  // Clear sparsity pattern data
  diagonal.clear();
  off_diagonal.clear();
  non_local.clear();
  full_rows.clear();

  // Check that primary dimension is valid
  if (_primary_dim > 1)
   {
    dolfin_error("SparsityPattern.cpp",
                 "primary dimension for sparsity pattern storage",
                 "Primary dimension must be less than 2 (0=row major, 1=column major");
  }

  const std::size_t local_size0
    = index_maps[_primary_dim]->size(IndexMap::MapSize::OWNED);

  const std::size_t primary_codim = _primary_dim == 0 ? 1 : 0;
  const std::size_t local_size1
    = index_maps[primary_codim]->size(IndexMap::MapSize::OWNED);
  const std::size_t global_size1
    = index_maps[primary_codim]->size(IndexMap::MapSize::GLOBAL);

  // Resize diagonal block
  diagonal.resize(local_size0);

  // Resize off-diagonal block (only needed when local range != global
  // range)
  if (global_size1 > local_size1)
  {
    dolfin_assert(_mpi_comm.size() > 1);
    off_diagonal.resize(local_size0);
  }
  else
  {
    dolfin_assert(global_size1 == local_size1);
    // NOTE: MPI::size(_mpi_comm)==1 does not necessarilly hold. It may be
    //       rectangle tensor with one very small dimension, e.g. Real space
  }
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert_global(dolfin::la_index i, dolfin::la_index j)
{
  const std::vector<ArrayView<const dolfin::la_index>> entries =
    { ArrayView<const dolfin::la_index>(1, &i),
      ArrayView<const dolfin::la_index>(1, &j)};

  insert_global(entries);
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert_global(
  const std::vector<ArrayView<const dolfin::la_index>>& entries)
{
  dolfin_assert(entries.size() == 2);

  // The primary_dim is global and must be mapped to local
  const auto primary_dim_map
      = [](const dolfin::la_index i_index, const IndexMap& index_map0)
    {
      dolfin_assert(index_map0.local_range().first <= (std::size_t) i_index
                    && (std::size_t) i_index < index_map0.local_range().second);
      return i_index - (dolfin::la_index) index_map0.local_range().first;
    };

  // The primary_codim is already global and stays the same
  const auto primary_codim_map
    = [](const dolfin::la_index j_index, const IndexMap& index_map1)
    { return j_index; };

  insert_entries(entries, primary_dim_map, primary_codim_map);
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert_local(
  const std::vector<ArrayView<const dolfin::la_index>>& entries)
{
  dolfin_assert(entries.size() == 2);

  // The primary_dim is local and stays the same
  const auto primary_dim_map
    = [](const dolfin::la_index i_index, const IndexMap& index_map0)
    { return i_index; };

  // The primary_codim must be mapped to global entries
  const auto primary_codim_map
    = [](const dolfin::la_index j_index,
         const IndexMap& index_map1) -> dolfin::la_index
    { return index_map1.local_to_global((std::size_t) j_index); };

  insert_entries(entries, primary_dim_map, primary_codim_map);
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert_local_global(
    const std::vector<ArrayView<const dolfin::la_index>>& entries)
{
  dolfin_assert(entries.size() == 2);

  // The primary_dim is local and stays the same
  const auto primary_dim_map
    = [](const dolfin::la_index i_index, const IndexMap& index_map0)
    { return i_index; };

  // The primary_codim is global and stays the same
  const auto primary_codim_map
    = [](const dolfin::la_index j_index, const IndexMap& index_map1)
    { return j_index; };

  insert_entries(entries, primary_dim_map, primary_codim_map);
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert_entries(
    const std::vector<ArrayView<const dolfin::la_index>>& entries,
    const std::function<dolfin::la_index(const dolfin::la_index, const IndexMap&)>& primary_dim_map,
    const std::function<dolfin::la_index(const dolfin::la_index, const IndexMap&)>& primary_codim_map)
{
  dolfin_assert(entries.size() == 2);
  const std::size_t _primary_dim = primary_dim();
  dolfin_assert(_primary_dim < 2);
  const std::size_t primary_codim = (_primary_dim + 1) % 2;
  dolfin_assert(primary_codim < 2);

  ArrayView<const dolfin::la_index> map_i = entries[_primary_dim];
  ArrayView<const dolfin::la_index> map_j = entries[primary_codim];

  const IndexMap& index_map0 = *_index_maps[ _primary_dim];
  const IndexMap& index_map1 = *_index_maps[primary_codim];
  const std::size_t local_size0 = index_map0.size(IndexMap::MapSize::OWNED);
  const auto& local_range1 = index_map1.local_range();

  const bool has_full_rows = full_rows.size() > 0;
  const auto full_rows_end = full_rows.end();

  // Programmers' note:
  // We use the lower case index i/j to denote the indices before calls to
  // primary_dim_map/primary_codim_map.
  // We use the  upper case index I/J to denote the indices after mapping
  // (using primary_dim_map/primary_codim_map) to be inserted into
  // the SparsityPattern data structure.
  //
  // In serial (_mpi_comm.size() == 1) we have the special case
  // where i == I and j == J.

  // Check local range
  if (_mpi_comm.size() == 1)
  {
    // Sequential mode, do simple insertion if not full row
    for (const auto &i_index : map_i)
    {
      dolfin_assert(i_index < (dolfin::la_index) diagonal.size());
      if (!has_full_rows || full_rows.find(i_index) == full_rows_end)
        diagonal[i_index].insert(map_j.begin(), map_j.end());
    }
  }
  else
  {
    // Parallel mode, use either diagonal, off_diagonal, non_local or
    // full_rows
    for (const auto &i_index : map_i)
    {
      const auto I = primary_dim_map(i_index, index_map0);
      // Full rows are stored separately
      if (has_full_rows && full_rows.find(I) != full_rows_end)
      {
        // Do nothing
        continue;
      }

      if (I < (dolfin::la_index) local_size0)
      {
        // Store local entry in diagonal or off-diagonal block
        for (const auto &j_index : map_j)
        {
          const auto J = primary_codim_map(j_index, index_map1);
          if ((dolfin::la_index) local_range1.first <= J
              && J < (dolfin::la_index) local_range1.second)
          {
            dolfin_assert(I < (dolfin::la_index) diagonal.size());
            diagonal[I].insert(J);
          }
          else
          {
            dolfin_assert(I < (dolfin::la_index) off_diagonal.size());
            off_diagonal[I].insert(J);
          }
        }
      }
      else
      {
        // Store non-local entry (communicated later during apply())
        for (const auto &j_index : map_j)
        {
          const auto J = primary_codim_map(j_index, index_map1);
          // Store indices
          non_local.push_back(I);
          non_local.push_back(J);
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert_full_rows_local(
  const std::vector<std::size_t>& rows)
{
  const std::size_t ghosted_size0 =
    _index_maps[_primary_dim]->size(IndexMap::MapSize::ALL);
  full_rows.set().reserve(rows.size());
  for (const auto row : rows)
  {
    dolfin_assert(row < ghosted_size0);
    full_rows.insert(row);
  }
}
//-----------------------------------------------------------------------------
std::size_t SparsityPattern::rank() const
{
  return 2;
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t>
  SparsityPattern::local_range(std::size_t dim) const
{
  dolfin_assert(dim < 2);
  return _index_maps[dim]->local_range();
}
//-----------------------------------------------------------------------------
std::size_t SparsityPattern::num_nonzeros() const
{
  std::size_t nz = 0;

  // Contribution from diagonal and off-diagonal
  for (const auto& slice : diagonal)
    nz += slice.size();
  for (const auto& slice : off_diagonal)
    nz += slice.size();

  // Contribution from full rows
  const std::size_t local_size0 =
    _index_maps[_primary_dim]->size(IndexMap::MapSize::OWNED);
  const std::size_t primary_codim = _primary_dim == 0 ? 1 : 0;
  const std::size_t ncols = _index_maps[primary_codim]->size(IndexMap::MapSize::GLOBAL);
  for (const auto& full_row : full_rows)
    if (full_row < local_size0)
      nz += ncols;

  return nz;
}
//-----------------------------------------------------------------------------
void SparsityPattern::num_nonzeros_diagonal(std::vector<std::size_t>& num_nonzeros) const
{
  // Resize vector
  num_nonzeros.resize(diagonal.size());

  // Get number of nonzeros per generalised row
  for (auto slice = diagonal.begin(); slice != diagonal.end(); ++slice)
    num_nonzeros[slice - diagonal.begin()] = slice->size();

  // Get number of nonzeros per full row
  if (full_rows.size() > 0)
  {
    const std::size_t local_size0 =
      _index_maps[_primary_dim]->size(IndexMap::MapSize::OWNED);
    const std::size_t primary_codim = _primary_dim == 0 ? 1 : 0;
    const std::size_t ncols = _index_maps[primary_codim]->size(IndexMap::MapSize::OWNED);
    for (const auto row : full_rows)
      if (row < local_size0)
        num_nonzeros[row] = ncols;
  }
}
//-----------------------------------------------------------------------------
void SparsityPattern::num_nonzeros_off_diagonal(std::vector<std::size_t>& num_nonzeros) const
{
  // Resize vector
  num_nonzeros.resize(off_diagonal.size());

  // Return if there is no off-diagonal
  if (off_diagonal.empty())
    return;

  // Compute number of nonzeros per generalised row
  for (auto slice = off_diagonal.begin(); slice != off_diagonal.end(); ++slice)
    num_nonzeros[slice - off_diagonal.begin()] = slice->size();

  // Get number of nonzeros per full row
  if (full_rows.size() > 0)
  {
    const std::size_t local_size0 =
      _index_maps[_primary_dim]->size(IndexMap::MapSize::OWNED);
    const std::size_t primary_codim = _primary_dim == 0 ? 1 : 0;
    const std::size_t ncols =
      _index_maps[primary_codim]->size(IndexMap::MapSize::GLOBAL)
      - _index_maps[primary_codim]->size(IndexMap::MapSize::OWNED);
    for (const auto row : full_rows)
      if (row < local_size0)
        num_nonzeros[row] = ncols;
  }
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
void SparsityPattern::apply()
{
  const std::size_t _primary_dim = primary_dim();
  const std::size_t primary_codim = (_primary_dim + 1) % 2;
  dolfin_assert(_primary_dim < 2);
  dolfin_assert(primary_codim < 2);

  const std::pair<dolfin::la_index, dolfin::la_index>
    local_range0 = _index_maps[_primary_dim]->local_range();
  const std::pair<dolfin::la_index, dolfin::la_index>
    local_range1 = _index_maps[primary_codim]->local_range();
  const std::size_t local_size0
    = _index_maps[_primary_dim]->size(IndexMap::MapSize::OWNED);
  const std::size_t offset0 = local_range0.first;

  const std::size_t num_processes = _mpi_comm.size();
  const std::size_t proc_number = _mpi_comm.rank();

  // Print some useful information
  if (get_log_level() <= DBG)
    info_statistics();

  // Communicate non-local blocks if any
  if (_mpi_comm.size() > 1)
  {
    // Figure out correct process for each non-local entry
    dolfin_assert(non_local.size() % 2 == 0);
    std::vector<std::vector<std::size_t>> non_local_send(num_processes);

    const std::vector<int>& off_process_owner
      = _index_maps[_primary_dim]->off_process_owner();

    const std::vector<std::size_t>& local_to_global
      = _index_maps[_primary_dim]->local_to_global_unowned();

    std::size_t dim_block_size = _index_maps[_primary_dim]->block_size();
    for (std::size_t i = 0; i < non_local.size(); i += 2)
    {
      // Get local indices of off-process dofs
      const std::size_t i_index = non_local[i];
      const std::size_t J = non_local[i + 1];

      // Figure out which process owns the row
      dolfin_assert(i_index >= local_size0);
      const std::size_t i_offset = (i_index - local_size0)/dim_block_size;
      dolfin_assert(i_offset < off_process_owner.size());
      const std::size_t p = off_process_owner[i_offset];

      dolfin_assert(p < num_processes);
      dolfin_assert(p != proc_number);

      // Get global I index
      la_index I = 0;
      if (i_index < local_size0)
        I = i_index + offset0;
      else
      {
        std::size_t tmp = i_index - local_size0;
        const std::div_t div = std::div((int) tmp, (int) dim_block_size);
        const int i_node = div.quot;
        const int i_component = div.rem;

        const std::size_t I_node = local_to_global[i_node];
        I = dim_block_size*I_node + i_component;
      }

      // Buffer local/global index pair to send
      non_local_send[p].push_back(I);
      non_local_send[p].push_back(J);
    }

    // Communicate non-local entries to other processes
    std::vector<std::size_t> non_local_received;
    MPI::all_to_all(_mpi_comm.comm(), non_local_send, non_local_received);

    // Insert non-local entries received from other processes
    dolfin_assert(non_local_received.size() % 2 == 0);

    for (std::size_t i = 0; i < non_local_received.size(); i += 2)
    {
      // Get global row and column
      const dolfin::la_index I = non_local_received[i];
      const dolfin::la_index J = non_local_received[i + 1];

      // Sanity check
      if (I < local_range0.first
          || I >= local_range0.second)
      {
        dolfin_error("SparsityPattern.cpp",
                     "apply changes to sparsity pattern",
                     "Received illegal sparsity pattern entry for row/column %d, not in range [%d, %d]",
                     I, local_range0.first,
                     local_range0.second);
      }

      // Get local I index
      const std::size_t i_index = I - offset0;

      // Insert in diagonal or off-diagonal block
      if (local_range1.first <= J &&
          J < local_range1.second)
      {
        dolfin_assert(i_index < diagonal.size());
        diagonal[i_index].insert(J);
      }
      else
      {
        dolfin_assert(i_index < off_diagonal.size());
        off_diagonal[i_index].insert(J);
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
  for (std::size_t i = 0; i < diagonal.size(); i++)
  {
    if (primary_dim() == 0)
      s << "Row " << i << ":";
    else
      s << "Col " << i << ":";

    for (const auto& entry : diagonal[i])
      s << " " << entry;

    if (!off_diagonal.empty())
    {
      for (const auto& entry : off_diagonal[i])
        s << " " << entry;
    }

    s << std::endl;
  }

  return s.str();
}
//-----------------------------------------------------------------------------
std::vector<std::vector<std::size_t>>
SparsityPattern::diagonal_pattern(Type type) const
{
  std::vector<std::vector<std::size_t>> v(diagonal.size());
  for (std::size_t i = 0; i < diagonal.size(); ++i)
    v[i].insert(v[i].begin(), diagonal[i].begin(), diagonal[i].end());

  if (type == Type::sorted)
  {
    for (std::size_t i = 0; i < v.size(); ++i)
      std::sort(v[i].begin(), v[i].end());
  }

  if (full_rows.size() > 0)
  {
    const std::size_t local_size0 =
      _index_maps[_primary_dim]->size(IndexMap::MapSize::OWNED);
    const std::size_t primary_codim = _primary_dim == 0 ? 1 : 0;
    const auto range1 = _index_maps[primary_codim]->local_range();
    for (const auto row : full_rows)
    {
      if (row >= local_size0)
        continue;
      dolfin_assert(v[row].size() == 0);
      v[row].reserve(range1.second - range1.first);
      for (std::size_t J = range1.first; J < range1.second; ++J)
        v[row].push_back(J);
    }
  }

  return v;
}
//-----------------------------------------------------------------------------
std::vector<std::vector<std::size_t>>
  SparsityPattern::off_diagonal_pattern(Type type) const
{
  std::vector<std::vector<std::size_t>> v(off_diagonal.size());
  for (std::size_t i = 0; i < off_diagonal.size(); ++i)
    v[i].insert(v[i].begin(), off_diagonal[i].begin(), off_diagonal[i].end());

  if (type == Type::sorted)
  {
    for (std::size_t i = 0; i < v.size(); ++i)
      std::sort(v[i].begin(), v[i].end());
  }

  if (full_rows.size() > 0)
  {
    const std::size_t local_size0 =
      _index_maps[_primary_dim]->size(IndexMap::MapSize::OWNED);
    const std::size_t primary_codim = _primary_dim == 0 ? 1 : 0;
    const auto range1 = _index_maps[primary_codim]->local_range();
    const std::size_t N1 = _index_maps[primary_codim]->size(IndexMap::MapSize::GLOBAL);
    for (const auto row : full_rows)
    {
      if (row >= local_size0)
        continue;
      dolfin_assert(v[row].size() == 0);
      v[row].reserve(N1 - (range1.second - range1.first));
      for (std::size_t J = 0; J < range1.first; ++J)
        v[row].push_back(J);
      for (std::size_t J = range1.second; J < N1; ++J)
        v[row].push_back(J);
    }
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
  const std::size_t num_nonzeros_non_local = non_local.size()/2;

  // Count total number of nonzeros
  const std::size_t num_nonzeros_total = num_nonzeros_diagonal
    + num_nonzeros_off_diagonal + num_nonzeros_non_local;

  std::size_t size0 = _index_maps[0]->size(IndexMap::MapSize::GLOBAL);
  std::size_t size1 = _index_maps[1]->size(IndexMap::MapSize::GLOBAL);

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
