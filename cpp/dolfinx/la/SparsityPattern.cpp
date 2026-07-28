// Copyright (C) 2007-2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "SparsityPattern.h"
#include <algorithm>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <limits>
#include <unordered_map>

using namespace dolfinx;
using namespace dolfinx::la;

namespace
{
/// @brief Bucket (row, column) entries by row.
///
/// Groups entries with matching row indices together, without
/// sorting or removing duplicates within a row. Used to obtain
/// per-row slices from the flat (row, column) insertion cache without
/// keeping one heap-allocated container per row.
///
/// @param[in] rows Row index of each entry.
/// @param[in] cols Column index of each entry (same length as `rows`).
/// @param[in] num_rows Number of rows to bucket into.
/// @return Row offsets (size `num_rows + 1`) and the column indices
/// grouped by row, i.e. row `i` occupies
/// `cols[offsets[i]:offsets[i + 1]]`.
std::pair<std::vector<std::int64_t>, std::vector<std::int32_t>>
bucket_by_row(std::span<const std::int32_t> rows,
              std::span<const std::int32_t> cols, std::int32_t num_rows)
{
  assert(rows.size() == cols.size());
  std::vector<std::int64_t> offsets(num_rows + 1, 0);
  for (std::int32_t row : rows)
    ++offsets[row + 1];
  std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());

  std::vector<std::int64_t> pos(offsets.begin(), std::prev(offsets.end()));
  std::vector<std::int32_t> bucketed(cols.size());
  for (std::size_t i = 0; i < rows.size(); ++i)
    bucketed[pos[rows[i]]++] = cols[i];

  return {std::move(offsets), std::move(bucketed)};
}
} // namespace

//-----------------------------------------------------------------------------
SparsityPattern::SparsityPattern(
    MPI_Comm comm, std::array<std::shared_ptr<const common::IndexMap>, 2> maps,
    std::array<int, 2> bs)
    : _comm(comm), _index_maps(std::move(maps)), _bs(bs)
{
  assert(_index_maps[0]);
}
//-----------------------------------------------------------------------------
SparsityPattern::SparsityPattern(
    MPI_Comm comm,
    const std::vector<std::vector<const SparsityPattern*>>& patterns,
    const std::array<std::vector<std::pair<
                         std::reference_wrapper<const common::IndexMap>, int>>,
                     2>& maps,
    const std::array<std::vector<int>, 2>& bs)
    : _comm(comm), _bs({1, 1})
{
  // FIXME: - Add range/bound checks for each block
  //        - Check for compatible block sizes for each block

  const auto [rank_offset0, local_offset0, ghosts_new0, owners0]
      = common::stack_index_maps(maps[0]);
  const auto [rank_offset1, local_offset1, ghosts_new1, owners1]
      = common::stack_index_maps(maps[1]);

  std::vector<std::int64_t> ghosts0, ghosts1;
  std::vector<std::int32_t> ghost_offsets0(1, 0);
  std::vector<std::int32_t> ghost_offsets1(1, 0);
  for (const std::vector<std::int64_t>& ghosts : ghosts_new0)
  {
    ghost_offsets0.push_back(ghost_offsets0.back() + ghosts.size());
    ghosts0.insert(ghosts0.end(), ghosts.begin(), ghosts.end());
  }
  for (const std::vector<std::int64_t>& ghosts : ghosts_new1)
  {
    ghost_offsets1.push_back(ghost_offsets1.back() + ghosts.size());
    ghosts1.insert(ghosts1.end(), ghosts.begin(), ghosts.end());
  }

  std::vector<int> ghost_owners0, ghost_owners1;
  for (const std::vector<int>& owners : owners0)
    ghost_owners0.insert(ghost_owners0.end(), owners.begin(), owners.end());
  for (const std::vector<int>& owners : owners1)
    ghost_owners1.insert(ghost_owners1.end(), owners.begin(), owners.end());

  // Create new IndexMaps
  _index_maps[0] = std::make_shared<common::IndexMap>(
      comm, local_offset0.back(), ghosts0, ghost_owners0);
  _index_maps[1] = std::make_shared<common::IndexMap>(
      comm, local_offset1.back(), ghosts1, ghost_owners1);

  const std::int32_t num_rows_local_new = _index_maps[0]->size_local();

  // Iterate over block rows
  for (std::size_t row = 0; row < patterns.size(); ++row)
  {
    const common::IndexMap& map_row = maps[0][row].first;
    const std::int32_t num_rows_local = map_row.size_local();
    const std::int32_t num_ghost_rows_local = map_row.num_ghosts();

    // Iterate over block columns of current row (block)
    for (std::size_t col = 0; col < patterns[row].size(); ++col)
    {
      const common::IndexMap& map_col = maps[1][col].first;
      const std::int32_t num_cols_local = map_col.size_local();
      // Get pattern for this block
      const SparsityPattern* p = patterns[row][col];
      if (!p)
        continue;

      if (!p->_offsets.empty())
      {
        throw std::runtime_error("Sub-sparsity pattern has been finalised. "
                                 "Cannot compute stacked pattern.");
      }

      const int bs_dof0 = bs[0][row];
      const int bs_dof1 = bs[1][col];

      // Bucket the sub-pattern's insertion cache by row once, giving
      // per-row column slices for the owned/ghost row loops below
      const auto [p_offsets, p_cols]
          = bucket_by_row(p->_cache_rows, p->_cache_cols,
                          num_rows_local + num_ghost_rows_local);

      // Iterate over owned rows cache
      for (std::int32_t i = 0; i < num_rows_local; ++i)
      {
        for (std::int64_t k = p_offsets[i]; k < p_offsets[i + 1]; ++k)
        {
          const std::int32_t c_old = p_cols[k];
          const std::int32_t r_new = bs_dof0 * i + local_offset0[row];
          const std::int32_t c_new = (c_old < num_cols_local)
                                         ? bs_dof1 * c_old + local_offset1[col]
                                         : bs_dof1 * (c_old - num_cols_local)
                                               + local_offset1.back()
                                               + ghost_offsets1[col];

          for (int k0 = 0; k0 < bs_dof0; ++k0)
          {
            for (int k1 = 0; k1 < bs_dof1; ++k1)
            {
              _cache_rows.push_back(r_new + k0);
              _cache_cols.push_back(c_new + k1);
            }
          }
        }
      }
      // Iterate over unowned rows cache
      for (std::int32_t i = 0; i < num_ghost_rows_local; ++i)
      {
        for (std::int64_t k = p_offsets[num_rows_local + i];
             k < p_offsets[num_rows_local + i + 1]; ++k)
        {
          const std::int32_t c_old = p_cols[k];
          const std::int32_t r_new = bs_dof0 * i + ghost_offsets0[row];
          const std::int32_t c_new = (c_old < num_cols_local)
                                         ? bs_dof1 * c_old + local_offset1[col]
                                         : bs_dof1 * (c_old - num_cols_local)
                                               + local_offset1.back()
                                               + ghost_offsets1[col];
          for (int k0 = 0; k0 < bs_dof0; ++k0)
          {
            for (int k1 = 0; k1 < bs_dof1; ++k1)
            {
              _cache_rows.push_back(num_rows_local_new + r_new + k0);
              _cache_cols.push_back(c_new + k1);
            }
          }
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert(std::int32_t row, std::int32_t col)
{
  if (!_offsets.empty())
  {
    throw std::runtime_error(
        "Cannot insert into sparsity pattern. It has already been finalized");
  }

  assert(_index_maps[0]);
  const std::int32_t max_row
      = _index_maps[0]->size_local() + _index_maps[0]->num_ghosts() - 1;

  if (row > max_row or row < 0)
  {
    throw std::runtime_error(
        "Cannot insert rows that do not exist in the IndexMap.");
  }

  _cache_rows.push_back(row);
  _cache_cols.push_back(col);
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert(std::span<const std::int32_t> rows,
                             std::span<const std::int32_t> cols)
{
  if (!_offsets.empty())
  {
    throw std::runtime_error(
        "Cannot insert into sparsity pattern. It has already been finalized");
  }

  assert(_index_maps[0]);
  const std::int32_t max_row
      = _index_maps[0]->size_local() + _index_maps[0]->num_ghosts() - 1;

  _cache_rows.reserve(_cache_rows.size() + rows.size() * cols.size());
  _cache_cols.reserve(_cache_cols.size() + rows.size() * cols.size());
  for (std::int32_t row : rows)
  {
    if (row > max_row or row < 0)
    {
      throw std::runtime_error(
          "Cannot insert rows that do not exist in the IndexMap.");
    }
    _cache_rows.insert(_cache_rows.end(), cols.size(), row);
    _cache_cols.insert(_cache_cols.end(), cols.begin(), cols.end());
  }
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert_diagonal(std::span<const std::int32_t> rows)
{
  if (!_offsets.empty())
  {
    throw std::runtime_error(
        "Cannot insert into sparsity pattern. It has already been finalized");
  }

  assert(_index_maps[0]);
  const std::int32_t max_row
      = _index_maps[0]->size_local() + _index_maps[0]->num_ghosts() - 1;

  for (std::int32_t row : rows)
  {
    if (row > max_row or row < 0)
    {
      throw std::runtime_error(
          "Cannot insert rows that do not exist in the IndexMap.");
    }
  }

  _cache_rows.insert(_cache_rows.end(), rows.begin(), rows.end());
  _cache_cols.insert(_cache_cols.end(), rows.begin(), rows.end());
}
//-----------------------------------------------------------------------------
std::shared_ptr<const common::IndexMap>
SparsityPattern::index_map(int dim) const
{
  return _index_maps.at(dim);
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t> SparsityPattern::column_indices() const
{
  if (_offsets.empty())
    throw std::runtime_error("Sparsity pattern has not been finalised.");

  std::array range = _index_maps[1]->local_range();
  const std::int32_t local_size = range[1] - range[0];
  const std::int32_t num_ghosts = static_cast<std::int32_t>(_col_ghosts.size());
  std::vector<std::int64_t> global(local_size + num_ghosts);
  std::iota(global.begin(), std::next(global.begin(), local_size), range[0]);
  std::ranges::copy(_col_ghosts, global.begin() + local_size);
  return global;
}
//-----------------------------------------------------------------------------
int SparsityPattern::block_size(int dim) const { return _bs[dim]; }
//-----------------------------------------------------------------------------
void SparsityPattern::finalize()
{
  if (!_offsets.empty())
    throw std::runtime_error("Sparsity pattern has already been finalised.");

  common::Timer t0("SparsityPattern::finalize");

  assert(_index_maps[0]);
  const std::int32_t local_size0 = _index_maps[0]->size_local();
  const std::array local_range0 = _index_maps[0]->local_range();
  std::span ghosts0 = _index_maps[0]->ghosts();
  std::span owners0 = _index_maps[0]->owners();
  std::span src0 = _index_maps[0]->src();

  assert(_index_maps[1]);
  const std::int32_t local_size1 = _index_maps[1]->size_local();
  const std::array local_range1 = _index_maps[1]->local_range();

  _col_ghosts.assign(_index_maps[1]->ghosts().begin(),
                     _index_maps[1]->ghosts().end());
  _col_ghost_owners.assign(_index_maps[1]->owners().begin(),
                           _index_maps[1]->owners().end());

  // Bucket the flat (row, column) insertion cache by row, giving
  // per-row slices without keeping one heap-allocated vector per row
  const std::int32_t num_rows0 = local_size0 + _index_maps[0]->num_ghosts();
  const auto [cache_offsets, cache_cols]
      = bucket_by_row(_cache_rows, _cache_cols, num_rows0);

  // Neighbourhood rank of the owner of each ghost row (looked up once
  // and reused below, rather than repeating the search per use)
  std::vector<int> neighbour_rank(owners0.size());
  std::ranges::transform(owners0, neighbour_rank.begin(),
                         [src0](int owner)
                         {
                           auto it = std::ranges::lower_bound(src0, owner);
                           assert(it != src0.end() and *it == owner);
                           return static_cast<int>(
                               std::distance(src0.begin(), it));
                         });

  // Compute size of data to send to each process
  std::vector<int> send_sizes(src0.size(), 0);
  for (std::size_t i = 0; i < owners0.size(); ++i)
  {
    // Guard against overflowing the int MPI count
    const std::size_t count = 3
                              * (cache_offsets[local_size0 + i + 1]
                                 - cache_offsets[local_size0 + i]);
    assert(send_sizes[neighbour_rank[i]] + count
           <= static_cast<std::size_t>(std::numeric_limits<int>::max()));
    send_sizes[neighbour_rank[i]] += count;
  }

  // Compute send displacements
  std::vector<int> send_disp(send_sizes.size() + 1, 0);
  std::partial_sum(send_sizes.begin(), send_sizes.end(),
                   std::next(send_disp.begin(), 1));

  // For each ghost row, pack and send (global row, global col,
  // col_owner) triplets to send to neighborhood
  std::vector<int> insert_pos(send_disp);
  std::vector<std::int64_t> ghost_data(send_disp.back());
  const int rank = dolfinx::MPI::rank(_comm.comm());
  for (std::size_t i = 0; i < owners0.size(); ++i)
  {
    for (std::int64_t k = cache_offsets[local_size0 + i];
         k < cache_offsets[local_size0 + i + 1]; ++k)
    {
      const std::int32_t col_local = cache_cols[k];

      // Get index in send buffer
      const std::int32_t pos = insert_pos[neighbour_rank[i]];

      // Pack send data
      ghost_data[pos] = ghosts0[i];
      if (col_local < local_size1)
      {
        ghost_data[pos + 1] = col_local + local_range1[0];
        ghost_data[pos + 2] = rank;
      }
      else
      {
        ghost_data[pos + 1] = _col_ghosts[col_local - local_size1];
        ghost_data[pos + 2] = _col_ghost_owners[col_local - local_size1];
      }

      insert_pos[neighbour_rank[i]] += 3;
    }
  }

  // Exchange data between processes
  std::vector<std::int64_t> ghost_data_in;
  {
    MPI_Comm comm;
    std::span dest0 = _index_maps[0]->dest();
    MPI_Dist_graph_create_adjacent(
        _index_maps[0]->comm(), dest0.size(), dest0.data(), MPI_UNWEIGHTED,
        src0.size(), src0.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm);

    std::vector<int> recv_sizes(dest0.size());
    send_sizes.reserve(1);
    recv_sizes.reserve(1);
    MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1,
                          MPI_INT, comm);

    // Build recv displacements
    std::vector<int> recv_disp{0};
    std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                     std::back_inserter(recv_disp));

    ghost_data_in.resize(recv_disp.back());
    MPI_Neighbor_alltoallv(ghost_data.data(), send_sizes.data(),
                           send_disp.data(), MPI_INT64_T, ghost_data_in.data(),
                           recv_sizes.data(), recv_disp.data(), MPI_INT64_T,
                           comm);
    MPI_Comm_free(&comm);
  }

  // Global to local map for ghost column indices
  std::unordered_map<std::int64_t, std::int32_t> global_to_local;
  global_to_local.reserve(_col_ghosts.size());
  std::int32_t local_i = local_size1;
  for (std::int64_t global_i : _col_ghosts)
    global_to_local.insert({global_i, local_i++});

  // Add data received from the neighborhood. Collected separately
  // from the local cache (only owned rows can receive) and bucketed
  // by row below, rather than appending into per-row vectors.
  std::vector<std::int32_t> recv_rows, recv_cols;
  recv_rows.reserve(ghost_data_in.size() / 3);
  recv_cols.reserve(ghost_data_in.size() / 3);
  for (std::size_t i = 0; i < ghost_data_in.size(); i += 3)
  {
    const std::int32_t row_local = ghost_data_in[i] - local_range0[0];
    const std::int64_t col = ghost_data_in[i + 1];
    const int owner = ghost_data_in[i + 2];
    recv_rows.push_back(row_local);
    if (col >= local_range1[0] and col < local_range1[1])
    {
      // Convert to local column index
      const std::int32_t J = col - local_range1[0];
      recv_cols.push_back(J);
    }
    else
    {
      // Column index may not exist in column indexmap
      auto it = global_to_local.insert({col, local_i});
      if (it.second)
      {
        _col_ghosts.push_back(col);
        _col_ghost_owners.push_back(owner);
        ++local_i;
      }

      recv_cols.push_back(it.first->second);
    }
  }
  const auto [recv_offsets, recv_cols_bucketed]
      = bucket_by_row(recv_rows, recv_cols, local_size0);

  // Reserve the (exact, pre-duplicate-removal) edge count so the edge
  // list is not repeatedly reallocated in the loop below
  _edges.reserve(cache_cols.size() + recv_cols_bucketed.size());

  // Sort and remove duplicate column indices in each row, building the
  // CSR offsets as we go. Offsets are std::int64_t to avoid
  // overflowing the running sum into _offsets. A single scratch buffer
  // is reused across rows rather than keeping one vector per row.
  _off_diagonal_offsets.resize(num_rows0);
  _offsets.reserve(num_rows0 + 1);
  _offsets.push_back(0);
  std::vector<std::int32_t> row;
  for (std::int32_t i = 0; i < num_rows0; ++i)
  {
    row.clear();
    row.insert(row.end(), cache_cols.begin() + cache_offsets[i],
               cache_cols.begin() + cache_offsets[i + 1]);
    if (i < local_size0)
    {
      row.insert(row.end(), recv_cols_bucketed.begin() + recv_offsets[i],
                 recv_cols_bucketed.begin() + recv_offsets[i + 1]);
    }

    std::ranges::sort(row);
    auto it_end = std::ranges::unique(row).begin();

    // Find position of first "off-diagonal" column
    _off_diagonal_offsets[i] = std::distance(
        row.begin(),
        std::ranges::lower_bound(row.begin(), it_end, local_size1));

    _edges.insert(_edges.end(), row.begin(), it_end);
    _offsets.push_back(_offsets.back() + std::distance(row.begin(), it_end));
  }

  // Clear cache
  std::vector<std::int32_t>().swap(_cache_rows);
  std::vector<std::int32_t>().swap(_cache_cols);

  _edges.shrink_to_fit();

  // Column count increased due to received rows from other processes
  spdlog::debug("Column ghost size increased from {} to {}",
                _index_maps[1]->ghosts().size(), _col_ghosts.size());

  // Update to new column index map
  _index_maps[1] = std::make_shared<common::IndexMap>(
      _comm.comm(), _index_maps[1]->size_local(), _col_ghosts,
      _col_ghost_owners);
}
//-----------------------------------------------------------------------------
std::int64_t SparsityPattern::num_nonzeros() const
{
  if (_offsets.empty())
    throw std::runtime_error("Sparsity pattern has not been finalized.");
  return _edges.size();
}
//-----------------------------------------------------------------------------
std::int32_t SparsityPattern::nnz_diag(std::int32_t row) const
{
  if (_offsets.empty())
    throw std::runtime_error("Sparsity pattern has not been finalized.");
  return _off_diagonal_offsets[row];
}
//-----------------------------------------------------------------------------
std::int32_t SparsityPattern::nnz_off_diag(std::int32_t row) const
{
  if (_offsets.empty())
    throw std::runtime_error("Sparsity pattern has not been finalized.");
  return (_offsets[row + 1] - _offsets[row]) - _off_diagonal_offsets[row];
}
//-----------------------------------------------------------------------------
std::pair<std::span<const std::int32_t>, std::span<const std::int64_t>>
SparsityPattern::graph() const
{
  if (_offsets.empty())
    throw std::runtime_error("Sparsity pattern has not been finalized.");
  return {_edges, _offsets};
}
//-----------------------------------------------------------------------------
std::span<const std::int32_t> SparsityPattern::off_diagonal_offsets() const
{
  if (_offsets.empty())
    throw std::runtime_error("Sparsity pattern has not been finalized.");
  return _off_diagonal_offsets;
}
//-----------------------------------------------------------------------------
MPI_Comm SparsityPattern::comm() const { return _comm.comm(); }
//-----------------------------------------------------------------------------
