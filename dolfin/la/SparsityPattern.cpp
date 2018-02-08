// Copyright (C) 2007-2011 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "SparsityPattern.h"
#include <algorithm>
#include <dolfin/common/MPI.h>
#include <dolfin/la/IndexMap.h>
#include <dolfin/log/LogStream.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
SparsityPattern::SparsityPattern(MPI_Comm comm, std::size_t primary_dim)
    : _primary_dim(primary_dim), _mpi_comm(comm)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void SparsityPattern::init(
    const std::array<std::shared_ptr<const IndexMap>, 2> index_maps,
    Ghosts ghosted)
{
  // Store index maps and ghosting
  _index_maps = index_maps;
  _ghosted = ghosted;

  const std::size_t _primary_dim = primary_dim();

  // Clear sparsity pattern data
  _diagonal.clear();
  _off_diagonal.clear();
  _non_local.clear();
  _full_rows.clear();

  // Check that primary dimension is valid
  if (_primary_dim > 1)
  {
    dolfin_error(
        "SparsityPattern.cpp", "primary dimension for sparsity pattern storage",
        "Primary dimension must be less than 2 (0=row major, 1=column major");
  }

  const std::size_t local_size0
      = index_maps[_primary_dim]->block_size()
        * index_maps[_primary_dim]->size(IndexMap::MapSize::OWNED);

  const std::size_t primary_codim = _primary_dim == 0 ? 1 : 0;
  const std::size_t local_size1
      = index_maps[primary_codim]->block_size()
        * index_maps[primary_codim]->size(IndexMap::MapSize::OWNED);
  const std::size_t global_size1
      = index_maps[primary_codim]->block_size()
        * index_maps[primary_codim]->size(IndexMap::MapSize::GLOBAL);

  // Resize diagonal block
  _diagonal.resize(local_size0);

  // Resize off-diagonal block (only needed when local range != global
  // range)
  if (global_size1 > local_size1)
  {
    dolfin_assert(_mpi_comm.size() > 1);
    _off_diagonal.resize(local_size0);
  }
  else
  {
    dolfin_assert(global_size1 == local_size1);
    // NOTE: MPI::size(_mpi_comm)==1 does not necessarilly hold. It may be
    //       rectangle tensor with one very small dimension, e.g. Real space
  }
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert_global(
    const std::array<ArrayView<const dolfin::la_index_t>, 2>& entries)
{
  // The primary_dim is global and must be mapped to local
  const auto primary_dim_map
      = [](const dolfin::la_index_t i_index,
           const IndexMap& index_map0) -> dolfin::la_index_t {
    std::size_t bs = index_map0.block_size();
    dolfin_assert(bs * index_map0.local_range()[0] <= (std::size_t)i_index
                  and (std::size_t) i_index < bs * index_map0.local_range()[1]);
    return i_index - (dolfin::la_index_t)bs * index_map0.local_range()[0];
  };

  // The primary_codim is already global and stays the same
  const auto primary_codim_map =
      [](const dolfin::la_index_t j_index,
         const IndexMap& index_map1) -> dolfin::la_index_t { return j_index; };

  insert_entries(entries, primary_dim_map, primary_codim_map);
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert_local(
    const std::array<ArrayView<const dolfin::la_index_t>, 2>& entries)
{
  // The primary_dim is local and stays the same
  const auto primary_dim_map =
      [](const dolfin::la_index_t i_index,
         const IndexMap& index_map0) -> dolfin::la_index_t { return i_index; };

  // The primary_codim must be mapped to global entries
  const auto primary_codim_map
      = [](const dolfin::la_index_t j_index,
           const IndexMap& index_map1) -> dolfin::la_index_t {
    return index_map1.local_to_global_index((std::size_t)j_index);
  };

  insert_entries(entries, primary_dim_map, primary_codim_map);
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert_local_global(
    const std::array<ArrayView<const dolfin::la_index_t>, 2>& entries)
{
  dolfin_assert(entries.size() == 2);

  // The primary_dim is local and stays the same
  const auto primary_dim_map =
      [](const dolfin::la_index_t i_index,
         const IndexMap& index_map0) -> dolfin::la_index_t { return i_index; };

  // The primary_codim is global and stays the same
  const auto primary_codim_map =
      [](const dolfin::la_index_t j_index,
         const IndexMap& index_map1) -> dolfin::la_index_t { return j_index; };

  insert_entries(entries, primary_dim_map, primary_codim_map);
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert_entries(
    const std::array<ArrayView<const dolfin::la_index_t>, 2>& entries,
    const std::function<dolfin::la_index_t(const dolfin::la_index_t,
                                           const IndexMap&)>& primary_dim_map,
    const std::function<dolfin::la_index_t(const dolfin::la_index_t,
                                           const IndexMap&)>& primary_codim_map)
{
  const std::size_t _primary_dim = primary_dim();
  dolfin_assert(_primary_dim < 2);
  const std::size_t primary_codim = (_primary_dim + 1) % 2;
  dolfin_assert(primary_codim < 2);

  ArrayView<const dolfin::la_index_t> map_i = entries[_primary_dim];
  ArrayView<const dolfin::la_index_t> map_j = entries[primary_codim];
  const IndexMap& index_map0 = *_index_maps[_primary_dim];
  const IndexMap& index_map1 = *_index_maps[primary_codim];

  std::size_t bs0 = index_map0.block_size();
  const std::size_t local_size0
      = bs0 * index_map0.size(IndexMap::MapSize::OWNED);

  std::size_t bs1 = index_map1.block_size();
  const auto local_range1 = index_map1.local_range();

  const bool has_full_rows = _full_rows.size() > 0;
  const auto full_rows_end = _full_rows.end();

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
    for (const auto& i_index : map_i)
    {
      dolfin_assert(i_index < (dolfin::la_index_t)_diagonal.size());
      if (!has_full_rows || _full_rows.find(i_index) == full_rows_end)
        _diagonal[i_index].insert(map_j.begin(), map_j.end());
    }
  }
  else
  {
    // Parallel mode, use either diagonal, off_diagonal, non_local or
    // full_rows
    for (const auto& i_index : map_i)
    {
      const auto I = primary_dim_map(i_index, index_map0);
      // Full rows are stored separately
      if (has_full_rows && _full_rows.find(I) != full_rows_end)
      {
        // Do nothing
        continue;
      }

      if (I < (dolfin::la_index_t)local_size0)
      {
        // Store local entry in diagonal or off-diagonal block
        for (const auto& j_index : map_j)
        {
          const auto J = primary_codim_map(j_index, index_map1);
          if ((dolfin::la_index_t)(bs1 * local_range1[0]) <= J
              and J < (dolfin::la_index_t)(bs1 * local_range1[1]))
          {
            dolfin_assert(I < (dolfin::la_index_t)_diagonal.size());
            _diagonal[I].insert(J);
          }
          else
          {
            dolfin_assert(I < (dolfin::la_index_t)_off_diagonal.size());
            _off_diagonal[I].insert(J);
          }
        }
      }
      else
      {
        // Store non-local entry (communicated later during apply())
        for (const auto& j_index : map_j)
        {
          const auto J = primary_codim_map(j_index, index_map1);
          // Store indices
          _non_local.push_back(I);
          _non_local.push_back(J);
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert_full_rows_local(
    const std::vector<std::size_t>& rows)
{
  std::size_t bs0 = _index_maps[_primary_dim]->block_size();
  const std::size_t ghosted_size0
      = bs0 * _index_maps[_primary_dim]->size(IndexMap::MapSize::ALL);
  _full_rows.set().reserve(rows.size());
  for (const auto row : rows)
  {
    dolfin_assert(row < ghosted_size0);
    _full_rows.insert(row);
  }
}
//-----------------------------------------------------------------------------
std::array<std::size_t, 2> SparsityPattern::local_range(std::size_t dim) const
{
  dolfin_assert(dim < 2);
  std::size_t bs = _index_maps[dim]->block_size();
  auto lrange = _index_maps[dim]->local_range();
  return {{bs * lrange[0], bs * lrange[1]}};
}
//-----------------------------------------------------------------------------
std::size_t SparsityPattern::num_nonzeros() const
{
  std::size_t nz = 0;

  // Contribution from diagonal and off-diagonal
  for (const auto& slice : _diagonal)
    nz += slice.size();
  for (const auto& slice : _off_diagonal)
    nz += slice.size();

  // Contribution from full rows
  std::size_t bs0 = _index_maps[_primary_dim]->block_size();
  const std::size_t local_size0
      = bs0 * _index_maps[_primary_dim]->size(IndexMap::MapSize::OWNED);

  const std::size_t primary_codim = _primary_dim == 0 ? 1 : 0;
  std::size_t bs1 = _index_maps[primary_codim]->block_size();
  const std::size_t ncols
      = bs1 * _index_maps[primary_codim]->size(IndexMap::MapSize::GLOBAL);
  for (const auto& full_row : _full_rows)
    if (full_row < local_size0)
      nz += ncols;

  return nz;
}
//-----------------------------------------------------------------------------
void SparsityPattern::num_nonzeros_diagonal(
    std::vector<std::size_t>& num_nonzeros) const
{
  // Resize vector
  num_nonzeros.resize(_diagonal.size());

  // Get number of nonzeros per generalised row
  for (auto slice = _diagonal.begin(); slice != _diagonal.end(); ++slice)
    num_nonzeros[slice - _diagonal.begin()] = slice->size();

  // Get number of nonzeros per full row
  if (_full_rows.size() > 0)
  {
    std::size_t bs0 = _index_maps[_primary_dim]->block_size();
    const std::size_t local_size0
        = bs0 * _index_maps[_primary_dim]->size(IndexMap::MapSize::OWNED);

    const std::size_t primary_codim = _primary_dim == 0 ? 1 : 0;
    std::size_t bs1 = _index_maps[primary_codim]->block_size();
    const std::size_t ncols
        = bs1 * _index_maps[primary_codim]->size(IndexMap::MapSize::OWNED);
    for (const auto row : _full_rows)
      if (row < local_size0)
        num_nonzeros[row] = ncols;
  }
}
//-----------------------------------------------------------------------------
void SparsityPattern::num_nonzeros_off_diagonal(
    std::vector<std::size_t>& num_nonzeros) const
{
  // Resize vector
  num_nonzeros.resize(_off_diagonal.size());

  // Return if there is no off-diagonal
  if (_off_diagonal.empty())
    return;

  // Compute number of nonzeros per generalised row
  for (auto slice = _off_diagonal.begin(); slice != _off_diagonal.end();
       ++slice)
    num_nonzeros[slice - _off_diagonal.begin()] = slice->size();

  // Get number of nonzeros per full row
  if (_full_rows.size() > 0)
  {
    std::size_t bs0 = _index_maps[_primary_dim]->block_size();
    const std::size_t local_size0
        = bs0 * _index_maps[_primary_dim]->size(IndexMap::MapSize::OWNED);

    const std::size_t primary_codim = _primary_dim == 0 ? 1 : 0;
    std::size_t bs1 = _index_maps[primary_codim]->block_size();
    const std::size_t ncols
        = bs1 * _index_maps[primary_codim]->size(IndexMap::MapSize::GLOBAL)
          - bs1 * _index_maps[primary_codim]->size(IndexMap::MapSize::OWNED);
    for (const auto row : _full_rows)
      if (row < local_size0)
        num_nonzeros[row] = ncols;
  }
}
//-----------------------------------------------------------------------------
void SparsityPattern::num_local_nonzeros(
    std::vector<std::size_t>& num_nonzeros) const
{
  num_nonzeros_diagonal(num_nonzeros);
  if (!_off_diagonal.empty())
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

  std::size_t bs0 = _index_maps[_primary_dim]->block_size();
  std::size_t bs1 = _index_maps[primary_codim]->block_size();
  const auto local_range0 = _index_maps[_primary_dim]->local_range();
  const auto local_range1 = _index_maps[primary_codim]->local_range();
  const std::size_t local_size0
      = bs0 * _index_maps[_primary_dim]->size(IndexMap::MapSize::OWNED);
  const std::size_t offset0 = bs0 * local_range0[0];

  const std::size_t num_processes = _mpi_comm.size();
  const std::size_t proc_number = _mpi_comm.rank();

  // Print some useful information
  if (get_log_level() <= DBG)
    info_statistics();

  // Communicate non-local blocks if any
  if (_mpi_comm.size() > 1)
  {
    // Figure out correct process for each non-local entry
    dolfin_assert(_non_local.size() % 2 == 0);
    std::vector<std::vector<std::size_t>> non_local_send(num_processes);

    const std::vector<int>& off_process_owner
        = _index_maps[_primary_dim]->block_off_process_owner();

    // Get local-to-global for unowned blocks
    const std::vector<std::size_t>& local_to_global
        = _index_maps[_primary_dim]->local_to_global_unowned();

    std::size_t dim_block_size = _index_maps[_primary_dim]->block_size();
    for (std::size_t i = 0; i < _non_local.size(); i += 2)
    {
      // Get local indices of off-process dofs
      const std::size_t i_index = _non_local[i];
      const std::size_t J = _non_local[i + 1];

      // Figure out which process owns the row
      dolfin_assert(i_index >= local_size0);
      const std::size_t i_offset = (i_index - local_size0) / dim_block_size;
      dolfin_assert(i_offset < off_process_owner.size());
      const std::size_t p = off_process_owner[i_offset];

      dolfin_assert(p < num_processes);
      dolfin_assert(p != proc_number);

      // Get global I index
      la_index_t I = 0;
      if (i_index < local_size0)
        I = i_index + offset0;
      else
      {
        std::size_t tmp = i_index - local_size0;
        const std::div_t div = std::div((int)tmp, (int)dim_block_size);
        const int i_node = div.quot;
        const int i_component = div.rem;

        const std::size_t I_node = local_to_global[i_node];
        I = dim_block_size * I_node + i_component;
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
      const dolfin::la_index_t I = non_local_received[i];
      const dolfin::la_index_t J = non_local_received[i + 1];

      // Sanity check
      if (I < local_range0[0]
          or I >= (dolfin::la_index_t)(bs0 * local_range0[1]))
      {
        dolfin_error("SparsityPattern.cpp", "apply changes to sparsity pattern",
                     "Received illegal sparsity pattern entry for row/column "
                     "%d, not in range [%d, %d]",
                     I, local_range0[0], local_range0[1]);
      }

      // Get local I index
      const std::size_t i_index = I - offset0;

      // Insert in diagonal or off-diagonal block
      if ((dolfin::la_index_t)(bs1 * local_range1[0]) <= J
          and J < (dolfin::la_index_t)(bs1 * local_range1[1]))
      {
        dolfin_assert(i_index < _diagonal.size());
        _diagonal[i_index].insert(J);
      }
      else
      {
        dolfin_assert(i_index < _off_diagonal.size());
        _off_diagonal[i_index].insert(J);
      }
    }
  }

  // Clear non-local entries
  _non_local.clear();
}
//-----------------------------------------------------------------------------
std::string SparsityPattern::str(bool verbose) const
{
  // Print each row
  std::stringstream s;
  for (std::size_t i = 0; i < _diagonal.size(); i++)
  {
    if (primary_dim() == 0)
      s << "Row " << i << ":";
    else
      s << "Col " << i << ":";

    for (const auto& entry : _diagonal[i])
      s << " " << entry;

    if (!_off_diagonal.empty())
    {
      for (const auto& entry : _off_diagonal[i])
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
  std::vector<std::vector<std::size_t>> v(_diagonal.size());
  for (std::size_t i = 0; i < _diagonal.size(); ++i)
    v[i].insert(v[i].begin(), _diagonal[i].begin(), _diagonal[i].end());

  if (type == Type::sorted)
  {
    for (std::size_t i = 0; i < v.size(); ++i)
      std::sort(v[i].begin(), v[i].end());
  }

  if (_full_rows.size() > 0)
  {
    std::size_t bs0 = _index_maps[_primary_dim]->block_size();
    const std::size_t local_size0
        = bs0 * _index_maps[_primary_dim]->size(IndexMap::MapSize::OWNED);

    const std::size_t primary_codim = _primary_dim == 0 ? 1 : 0;
    std::size_t bs1 = _index_maps[primary_codim]->block_size();
    const auto range1 = _index_maps[primary_codim]->local_range();
    for (const auto row : _full_rows)
    {
      if (row >= local_size0)
        continue;
      dolfin_assert(v[row].size() == 0);
      v[row].reserve(range1[1] - range1[0]);
      for (std::size_t J = bs1 * range1[0]; J < bs1 * range1[1]; ++J)
        v[row].push_back(J);
    }
  }

  return v;
}
//-----------------------------------------------------------------------------
std::vector<std::vector<std::size_t>>
SparsityPattern::off_diagonal_pattern(Type type) const
{
  std::vector<std::vector<std::size_t>> v(_off_diagonal.size());
  for (std::size_t i = 0; i < _off_diagonal.size(); ++i)
    v[i].insert(v[i].begin(), _off_diagonal[i].begin(), _off_diagonal[i].end());

  if (type == Type::sorted)
  {
    for (std::size_t i = 0; i < v.size(); ++i)
      std::sort(v[i].begin(), v[i].end());
  }

  if (_full_rows.size() > 0)
  {
    std::size_t bs0 = _index_maps[_primary_dim]->block_size();
    const std::size_t local_size0
        = bs0 * _index_maps[_primary_dim]->size(IndexMap::MapSize::OWNED);

    const std::size_t primary_codim = _primary_dim == 0 ? 1 : 0;
    std::size_t bs1 = _index_maps[primary_codim]->block_size();
    const auto range1 = _index_maps[primary_codim]->local_range();
    const std::size_t N1
        = bs1 * _index_maps[primary_codim]->size(IndexMap::MapSize::GLOBAL);
    for (const auto row : _full_rows)
    {
      if (row >= local_size0)
        continue;
      dolfin_assert(v[row].size() == 0);
      v[row].reserve(N1 - (range1[1] - range1[0]));
      for (std::size_t J = 0; J < bs1 * range1[0]; ++J)
        v[row].push_back(J);
      for (std::size_t J = bs1 * range1[1]; J < N1; ++J)
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
  for (std::size_t i = 0; i < _diagonal.size(); ++i)
    num_nonzeros_diagonal += _diagonal[i].size();

  // Count nonzeros in off-diagonal block
  std::size_t num_nonzeros_off_diagonal = 0;
  for (std::size_t i = 0; i < _off_diagonal.size(); ++i)
    num_nonzeros_off_diagonal += _off_diagonal[i].size();

  // Count nonzeros in non-local block
  const std::size_t num_nonzeros_non_local = _non_local.size() / 2;

  // Count total number of nonzeros
  const std::size_t num_nonzeros_total = num_nonzeros_diagonal
                                         + num_nonzeros_off_diagonal
                                         + num_nonzeros_non_local;

  std::size_t bs0 = _index_maps[0]->block_size();
  std::size_t size0 = bs0 * _index_maps[0]->size(IndexMap::MapSize::GLOBAL);

  std::size_t bs1 = _index_maps[1]->block_size();
  std::size_t size1 = bs1 * _index_maps[1]->size(IndexMap::MapSize::GLOBAL);

  // Return number of entries
  cout << "Matrix of size " << size0 << " x " << size1 << " has "
       << num_nonzeros_total << " ("
       << 100.0 * num_nonzeros_total / (size0 * size1) << "%)"
       << " nonzero entries." << endl;
  if (num_nonzeros_total != num_nonzeros_diagonal)
  {
    cout << "Diagonal: " << num_nonzeros_diagonal << " ("
         << (100.0 * static_cast<double>(num_nonzeros_diagonal)
             / static_cast<double>(num_nonzeros_total))
         << "%), ";
    cout << "off-diagonal: " << num_nonzeros_off_diagonal << " ("
         << (100.0 * static_cast<double>(num_nonzeros_off_diagonal)
             / static_cast<double>(num_nonzeros_total))
         << "%), ";
    cout << "non-local: " << num_nonzeros_non_local << " ("
         << (100.0 * static_cast<double>(num_nonzeros_non_local)
             / static_cast<double>(num_nonzeros_total))
         << "%)";
    cout << endl;
  }
}
//-----------------------------------------------------------------------------
SparsityPattern::SparsityPattern(
    const std::vector<std::vector<const SparsityPattern*>> patterns,
    std::vector<std::int32_t> offsets0, std::vector<std::int32_t> offsets1)
    : _primary_dim(0), _mpi_comm(MPI_COMM_WORLD)
{
  // FIXME: - Extend for parallel
  //        - Add range/bound checks
  //        - support null blocks
  //        - Update IndexSets
  //        - Check for compatible block sizes

  // Sum local sizes
  std::size_t local_size0 = 0;
  for (std::size_t row = 0; row < patterns.size(); ++row)
  {
    assert(patterns[row][0]);
    local_size0
        += patterns[row][0]->_index_maps[0]->size(IndexMap::MapSize::OWNED);
  }

  std::size_t local_size1 = 0;
  for (std::size_t col = 0; col < patterns[0].size(); ++col)
  {
    assert(patterns[0][col]);
    local_size1
        += patterns[0][col]->_index_maps[1]->size(IndexMap::MapSize::OWNED);
  }

  assert(patterns[0][0]);
  _index_maps[0]
      = std::make_shared<IndexMap>(patterns[0][0]->mpi_comm(), local_size0, 1);
  _index_maps[1]
      = std::make_shared<IndexMap>(patterns[0][0]->mpi_comm(), local_size1, 1);

  // Merge sparsity patterns

  // Iterate over rows
  for (std::size_t row = 0; row < patterns.size(); ++row)
  {
    // std::cout << "Row: " << row << ", " << patterns.size() << std::endl;

    // Get offset for rows (nodes)
    const std::size_t row_offset = offsets0[row];

    // std::cout << "*** Row offset: " << row_offset << std::endl;

    // Iterate over columns of current row
    this->_diagonal.resize(this->_diagonal.size()
                           + patterns[row][0]->_diagonal.size());
    for (std::size_t col = 0; col < patterns[row].size(); ++col)
    {
      // FIXME: this need to be global
      // Get offset for columns (edges)
      std::size_t col_offset = offsets1[col];

      // std::cout << "  Col offset: " << col_offset << std::endl;

      // Iterate over nodes in sparsity pattern
      if (patterns[row][col])
      {
        for (std::size_t k = 0; k < patterns[row][col]->_diagonal.size(); ++k)
        {
          // std::cout << "    node: " << k << std::endl;

          // Get nodes edges, and add offset
          // std::cout << "Get edges" << std::endl;
          std::vector<std::size_t> edges
              = patterns[row][col]->_diagonal[k].set();

          // std::cout << "Add offset " << std::endl;
          // for (auto e : edges)
          //  std::cout << "Pre-edge: "<< e << std::endl;
          // std::cout << "Transform" << std::endl;
          std::transform(edges.begin(), edges.end(), edges.begin(),
                         std::bind2nd(std::plus<double>(), col_offset));
          // std::cout << "Add edges to pattern" << std::endl;
          // for (auto e : edges)
          //  std::cout << "Post-edge: "<< e << std::endl;
          // std::cout << "Insert into row: " << k + row_offset << std::endl;
          // std::cout << "Insert: " << k << ", " << row_offset << std::endl;
          assert(k + row_offset < this->_diagonal.size());
          this->_diagonal[k + row_offset].insert(edges.begin(), edges.end());
          // std::cout << "Post Insert" << std::endl;
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
