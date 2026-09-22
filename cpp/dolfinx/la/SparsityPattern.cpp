// Copyright (C) 2007-2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "SparsityPattern.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

using namespace dolfinx;
using namespace dolfinx::la;

namespace
{
/// @brief Bucket (row, column) entries by row (not sorted or deduped
/// within a row).
/// @param[in] rows Row index of each entry.
/// @param[in] cols Column index of each entry (same length as `rows`).
/// @param[in] num_rows Number of rows to bucket into.
/// @return Row offsets (size `num_rows + 1`) and column indices
/// grouped by row: row `i` occupies `cols[offsets[i]:offsets[i+1]]`.
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

/// @brief Group block indices by row, so that the blocks touching a row
/// can be visited without re-scanning the whole cache.
/// @param[in] rows Row indices of all blocks, concatenated.
/// @param[in] offsets Offsets of each block into `rows`, size
/// `num_blocks + 1`.
/// @param[in] num_rows Number of rows to group by.
/// @return Row offsets (size `num_rows + 1`) and block indices grouped
/// by row, or two empty vectors if the cache holds no blocks.
std::pair<std::vector<std::int64_t>, std::vector<std::int32_t>>
transpose_blocks(std::span<const std::int32_t> rows,
                 std::span<const std::int64_t> offsets, std::int32_t num_rows)
{
  if (rows.empty())
    return {};

  const std::size_t nblocks = offsets.size() - 1;
  assert(nblocks
         <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()));

  std::vector<std::int64_t> block_offsets(num_rows + 1, 0);
  for (std::int32_t row : rows)
    ++block_offsets[row + 1];
  std::partial_sum(block_offsets.begin(), block_offsets.end(),
                   block_offsets.begin());

  std::vector<std::int32_t> block_ids(block_offsets.back());
  std::vector<std::int64_t> pos(block_offsets.begin(),
                                std::prev(block_offsets.end()));
  for (std::size_t b = 0; b < nblocks; ++b)
  {
    for (std::int64_t i = offsets[b]; i < offsets[b + 1]; ++i)
      block_ids[pos[rows[i]]++] = static_cast<std::int32_t>(b);
  }

  return {std::move(block_offsets), std::move(block_ids)};
}

/// @brief As transpose_blocks, but for blocks that all hold `bs` row
/// indices, so that block `b` occupies `rows[b * bs : (b + 1) * bs]`.
/// @param[in] rows Row indices of all blocks, concatenated.
/// @param[in] bs Number of row indices in each block.
/// @param[in] num_rows Number of rows to group by.
/// @return Row offsets (size `num_rows + 1`) and block indices grouped
/// by row, or two empty vectors if the cache holds no blocks.
std::pair<std::vector<std::int64_t>, std::vector<std::int32_t>>
transpose_blocks_uniform(std::span<const std::int32_t> rows, std::int32_t bs,
                         std::int32_t num_rows)
{
  if (rows.empty())
    return {};

  assert(bs > 0);
  const std::size_t nblocks = rows.size() / bs;
  assert(nblocks
         <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()));

  std::vector<std::int64_t> block_offsets(num_rows + 1, 0);
  for (std::int32_t row : rows)
    ++block_offsets[row + 1];
  std::partial_sum(block_offsets.begin(), block_offsets.end(),
                   block_offsets.begin());

  // The block index is tracked by a counter rather than computed as
  // i / bs, which would be a runtime integer division per entry.
  std::vector<std::int32_t> block_ids(block_offsets.back());
  std::vector<std::int64_t> pos(block_offsets.begin(),
                                std::prev(block_offsets.end()));
  for (std::size_t b = 0, i = 0; b < nblocks; ++b)
  {
    for (std::int32_t k = 0; k < bs; ++k, ++i)
      block_ids[pos[rows[i]]++] = static_cast<std::int32_t>(b);
  }

  return {std::move(block_offsets), std::move(block_ids)};
}
} // namespace

std::pair<std::vector<std::int64_t>, std::vector<std::int32_t>>
SparsityPattern::bucket_cache(std::int32_t num_rows,
                              std::int32_t num_cols) const
{
  // Group block indices by row. This avoids traversing every block entry
  // twice to determine the deduplicated output size.
  const auto [block_offsets, block_ids]
      = transpose_blocks(_cache_brows, _cache_boffs_r, num_rows);
  const auto [sblock_offsets, sblock_ids]
      = _cache_sbs > 0
            ? transpose_blocks_uniform(_cache_srows, _cache_sbs, num_rows)
            : transpose_blocks(_cache_srows, _cache_soffs, num_rows);

  const auto [pair_offsets, pair_cols]
      = bucket_by_row(_cache_rows, _cache_cols, num_rows);
  std::vector<bool> diagonal(num_rows, false);
  for (std::int32_t row : _cache_diag)
    diagonal[row] = true;

  // Traverse each block entry once and append unique columns row-wise.
  std::vector<std::int32_t> last_seen(num_cols, -1);
  std::vector<std::int64_t> offsets;
  offsets.reserve(num_rows + 1);
  offsets.push_back(0);
  std::vector<std::int32_t> bucketed;

  // The de-duplicated size is not known up front. Reserve the number of
  // cached column indices: an over-estimate where a column is shared by
  // many blocks, an under-estimate for wide blocks, but in both cases
  // far closer than growing from empty.
  bucketed.reserve(_cache_bcols.size() + _cache_srows.size()
                   + _cache_cols.size());

  const bool has_blocks = !block_offsets.empty();
  const bool has_sblocks = !sblock_offsets.empty();
  for (std::int32_t row = 0; row < num_rows; ++row)
  {
    if (has_blocks)
    {
      for (std::int64_t k = block_offsets[row]; k < block_offsets[row + 1]; ++k)
      {
        const std::int32_t b = block_ids[k];
        for (std::int64_t i = _cache_boffs_c[b]; i < _cache_boffs_c[b + 1]; ++i)
        {
          if (std::int32_t col = _cache_bcols[i]; last_seen[col] != row)
          {
            last_seen[col] = row;
            bucketed.push_back(col);
          }
        }
      }
    }
    if (has_sblocks and _cache_sbs > 0)
    {
      // Blocks of a single width: no offsets to load, and the inner
      // trip count is the same for every block
      const std::int64_t bs = _cache_sbs;
      for (std::int64_t k = sblock_offsets[row]; k < sblock_offsets[row + 1];
           ++k)
      {
        const std::int64_t begin = bs * sblock_ids[k];
        for (std::int64_t i = begin; i < begin + bs; ++i)
        {
          if (std::int32_t col = _cache_srows[i]; last_seen[col] != row)
          {
            last_seen[col] = row;
            bucketed.push_back(col);
          }
        }
      }
    }
    else if (has_sblocks)
    {
      for (std::int64_t k = sblock_offsets[row]; k < sblock_offsets[row + 1];
           ++k)
      {
        const std::int32_t b = sblock_ids[k];
        for (std::int64_t i = _cache_soffs[b]; i < _cache_soffs[b + 1]; ++i)
        {
          if (std::int32_t col = _cache_srows[i]; last_seen[col] != row)
          {
            last_seen[col] = row;
            bucketed.push_back(col);
          }
        }
      }
    }
    for (std::int64_t k = pair_offsets[row]; k < pair_offsets[row + 1]; ++k)
    {
      if (std::int32_t col = pair_cols[k]; last_seen[col] != row)
      {
        last_seen[col] = row;
        bucketed.push_back(col);
      }
    }
    if (diagonal[row] and last_seen[row] != row)
      bucketed.push_back(row);
    offsets.push_back(bucketed.size());
  }

  return {std::move(offsets), std::move(bucketed)};
}
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

      // Bucket the sub-pattern's cache by row for the loops below
      const auto [p_offsets, p_cols]
          = p->bucket_cache(num_rows_local + num_ghost_rows_local,
                            num_cols_local + map_col.num_ghosts());

      std::size_t num_blocks = 0;
      for (std::size_t i = 0; i + 1 < p_offsets.size(); ++i)
        num_blocks += p_offsets[i] != p_offsets[i + 1];
      reserve_blocks(num_blocks, num_blocks * bs_dof0, p_cols.size() * bs_dof1);

      // Store each expanded source row as one outer-product block.
      const auto append_row
          = [this, &p_offsets, &p_cols, &local_offset1, &ghost_offsets1, col,
             num_cols_local, bs_dof0,
             bs_dof1](std::int32_t old_row, std::int32_t new_row)
      {
        if (p_offsets[old_row] == p_offsets[old_row + 1])
          return;

        for (int k0 = 0; k0 < bs_dof0; ++k0)
          _cache_brows.push_back(new_row + k0);
        for (std::int64_t k = p_offsets[old_row]; k < p_offsets[old_row + 1];
             ++k)
        {
          const std::int32_t c_old = p_cols[k];
          const std::int32_t c_new = (c_old < num_cols_local)
                                         ? bs_dof1 * c_old + local_offset1[col]
                                         : bs_dof1 * (c_old - num_cols_local)
                                               + local_offset1.back()
                                               + ghost_offsets1[col];
          for (int k1 = 0; k1 < bs_dof1; ++k1)
            _cache_bcols.push_back(c_new + k1);
        }
        _cache_boffs_r.push_back(
            static_cast<std::int64_t>(_cache_brows.size()));
        _cache_boffs_c.push_back(
            static_cast<std::int64_t>(_cache_bcols.size()));
      };

      // Iterate over owned rows cache
      for (std::int32_t i = 0; i < num_rows_local; ++i)
      {
        const std::int32_t r_new = bs_dof0 * i + local_offset0[row];
        append_row(i, r_new);
      }

      // Iterate over unowned rows cache
      for (std::int32_t i = 0; i < num_ghost_rows_local; ++i)
      {
        const std::int32_t r_new
            = num_rows_local_new + bs_dof0 * i + ghost_offsets0[row];
        append_row(num_rows_local + i, r_new);
      }
    }
  }
}
//-----------------------------------------------------------------------------
void SparsityPattern::reserve_blocks(std::size_t num_blocks,
                                     std::size_t num_rows, std::size_t num_cols)
{
  if (!_offsets.empty())
  {
    throw std::runtime_error(
        "Cannot reserve in sparsity pattern. It has already been finalized");
  }

  // Which of the two block caches the blocks land in is only known once
  // insert() sees the spans. Reserving both maps around 2.5x the cache
  // that is actually filled, which costs more than the untouched pages
  // suggest, so hold the request and let insert() apply it to the
  // cache it uses.
  _reserve = {_reserve[0] + num_blocks, _reserve[1] + num_rows,
              _reserve[2] + num_cols};
}
//-----------------------------------------------------------------------------
void SparsityPattern::expand_square_offsets()
{
  assert(_cache_sbs > 0);
  const std::size_t nblocks = _cache_srows.size() / _cache_sbs;
  _cache_soffs.resize(nblocks + 1);
  for (std::size_t b = 0; b <= nblocks; ++b)
    _cache_soffs[b] = static_cast<std::int64_t>(b) * _cache_sbs;
  _cache_sbs = -1;
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

  // Store the block as inserted rather than expanding it to
  // rows.size() * cols.size() (row, column) pairs; finalize() expands
  // it. Note: no explicit reserve() here -- reserve(n) allocates
  // exactly n, not the amortised capacity insert() would pick, so
  // reserving on every call would be O(#insert calls^2).
  if (rows.data() == cols.data() and rows.size() == cols.size())
  {
    // Same index list for rows and columns, e.g. a cell block for a
    // form whose test and trial dofmaps coincide. Cache it once.
    if (_reserve[0] > 0)
    {
      _cache_srows.reserve(_cache_srows.size() + _reserve[1]);
      _reserve = {0, 0, 0};
    }

    // An empty block adds no entries, but caching it is not free: it
    // would count as a block of a different width and drop the square
    // cache from stride indexing to explicit offsets.
    // sparsitybuild::interior_facets passes an empty list for a facet
    // with no cell on one side, so this is a live path.
    const std::int32_t bs = static_cast<std::int32_t>(rows.size());
    if (bs == 0)
      return;

    if (_cache_sbs == 0)
      _cache_sbs = bs;
    else if (_cache_sbs > 0 and _cache_sbs != bs)
      expand_square_offsets();

    _cache_srows.insert(_cache_srows.end(), rows.begin(), rows.end());
    if (_cache_sbs < 0)
      _cache_soffs.push_back(static_cast<std::int64_t>(_cache_srows.size()));
  }
  else
  {
    if (_reserve[0] > 0)
    {
      _cache_brows.reserve(_cache_brows.size() + _reserve[1]);
      _cache_bcols.reserve(_cache_bcols.size() + _reserve[2]);
      _cache_boffs_r.reserve(_cache_boffs_r.size() + _reserve[0]);
      _cache_boffs_c.reserve(_cache_boffs_c.size() + _reserve[0]);
      _reserve = {0, 0, 0};
    }

    _cache_brows.insert(_cache_brows.end(), rows.begin(), rows.end());
    _cache_bcols.insert(_cache_bcols.end(), cols.begin(), cols.end());
    _cache_boffs_r.push_back(static_cast<std::int64_t>(_cache_brows.size()));
    _cache_boffs_c.push_back(static_cast<std::int64_t>(_cache_bcols.size()));
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
  _cache_diag.insert(_cache_diag.end(), rows.begin(), rows.end());
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

  // Bucket the insertion cache by row for the loops below
  const std::int32_t num_rows0 = local_size0 + _index_maps[0]->num_ghosts();
  const std::int32_t num_cols_cache
      = local_size1 + _index_maps[1]->num_ghosts();
  auto [cache_offsets, cache_cols] = bucket_cache(num_rows0, num_cols_cache);

  // Cache is now fully bucketed into cache_offsets/cache_cols; drop it
  // early to reduce peak memory for the rest of finalize()
  std::vector<std::int32_t>().swap(_cache_brows);
  std::vector<std::int32_t>().swap(_cache_bcols);
  std::vector<std::int64_t>().swap(_cache_boffs_r);
  std::vector<std::int64_t>().swap(_cache_boffs_c);
  std::vector<std::int32_t>().swap(_cache_srows);
  std::vector<std::int64_t>().swap(_cache_soffs);
  _cache_sbs = 0;
  std::vector<std::int32_t>().swap(_cache_rows);
  std::vector<std::int32_t>().swap(_cache_cols);
  std::vector<std::int32_t>().swap(_cache_diag);
  _reserve = {0, 0, 0};

  // Exchange ghost-row entries and bucket the received entries. Keep all
  // communication and mapping work arrays scoped so they are released before
  // the final graph is allocated.
  std::pair<std::vector<std::int64_t>, std::vector<std::int32_t>> recv_bucket;
  {
    // Neighbourhood rank of each ghost row's owner, looked up once
    std::vector<int> neighbour_rank(owners0.size());
    std::ranges::transform(owners0, neighbour_rank.begin(),
                           [src0](int owner)
                           {
                             auto it = std::ranges::lower_bound(src0, owner);
                             assert(it != src0.end() and *it == owner);
                             return static_cast<int>(
                                 std::ranges::distance(src0.begin(), it));
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
      MPI_Dist_graph_create_adjacent(_index_maps[0]->comm(), dest0.size(),
                                     dest0.data(), MPI_UNWEIGHTED, src0.size(),
                                     src0.data(), MPI_UNWEIGHTED, MPI_INFO_NULL,
                                     false, &comm);

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
                             send_disp.data(), MPI_INT64_T,
                             ghost_data_in.data(), recv_sizes.data(),
                             recv_disp.data(), MPI_INT64_T, comm);
      MPI_Comm_free(&comm);
    }

    // Global to local map for ghost column indices. Reserve for the
    // worst case where every received entry is a new ghost column, to
    // avoid rehashing while the map is populated below.
    std::unordered_map<std::int64_t, std::int32_t> global_to_local;
    global_to_local.reserve(_col_ghosts.size() + ghost_data_in.size() / 3);
    std::int32_t local_i = local_size1;
    for (std::int64_t global_i : _col_ghosts)
      global_to_local.insert({global_i, local_i++});

    // Add data received from the neighborhood, bucketed by row below
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
    recv_bucket = bucket_by_row(recv_rows, recv_cols, local_size0);
  }
  const auto& [recv_offsets, recv_cols_bucketed] = recv_bucket;

  _off_diagonal_offsets.resize(num_rows0);
  if (recv_cols_bucketed.empty())
  {
    // Nothing was received, so the de-duplicated cache is already the
    // graph. Adopt its storage rather than counting the edges and
    // copying each row into a second buffer of the same size.
    _offsets = std::move(cache_offsets);
    _edges = std::move(cache_cols);
    for (std::int32_t i = 0; i < num_rows0; ++i)
    {
      std::vector<std::int32_t>::iterator row_start
          = std::next(_edges.begin(), _offsets[i]);
      std::vector<std::int32_t>::iterator row_end
          = std::next(_edges.begin(), _offsets[i + 1]);
      std::ranges::sort(row_start, row_end);

      // Find position of first "off-diagonal" column
      _off_diagonal_offsets[i] = std::ranges::distance(
          row_start, std::ranges::lower_bound(row_start, row_end, local_size1));
    }
  }
  else
  {
    // De-duplicate each row's raw (unsorted, repeats included) column
    // list with a generation-stamped marker: last_seen[col] == i means
    // col has already been recorded for row i. Rows are visited exactly
    // once in increasing order, so the row index itself is the stamp --
    // no reset between rows needed. This turns per-row de-duplication
    // from O(m log m) (sort the raw, repeat-laden list) into
    // O(m + k log k), where k <= m is the de-duplicated count: sorting
    // only ever runs over the much smaller de-duplicated list.
    const std::int32_t num_cols0
        = local_size1 + static_cast<std::int32_t>(_col_ghosts.size());
    std::vector<std::int32_t> last_seen(num_cols0, -1);

    // Size of the union of the cached and the received entries
    std::size_t num_edges = 0;
    for (std::int32_t i = 0; i < num_rows0; ++i)
    {
      for (std::int64_t k = cache_offsets[i]; k < cache_offsets[i + 1]; ++k)
      {
        if (std::int32_t c = cache_cols[k]; last_seen[c] != i)
        {
          last_seen[c] = i;
          ++num_edges;
        }
      }
      if (i < local_size0)
      {
        for (std::int64_t k = recv_offsets[i]; k < recv_offsets[i + 1]; ++k)
        {
          if (std::int32_t c = recv_cols_bucketed[k]; last_seen[c] != i)
          {
            last_seen[c] = i;
            ++num_edges;
          }
        }
      }
    }
    std::ranges::fill(last_seen, -1);
    _edges.reserve(num_edges);

    // Build CSR offsets as we go. Offsets are int64_t to avoid overflow.
    // _edges is reserved to the exact edge count above, so appending in
    // place never reallocates.
    _offsets.reserve(num_rows0 + 1);
    _offsets.push_back(0);
    for (std::int32_t i = 0; i < num_rows0; ++i)
    {
      const std::size_t row_begin = _edges.size();
      if (i < local_size0 and recv_offsets[i] != recv_offsets[i + 1])
      {
        // bucket_cache() already de-duplicated the cached columns of this
        // row, so they only need stamping; the received columns are the
        // ones that may repeat them.
        for (std::int64_t k = cache_offsets[i]; k < cache_offsets[i + 1]; ++k)
        {
          const std::int32_t c = cache_cols[k];
          last_seen[c] = i;
          _edges.push_back(c);
        }
        for (std::int64_t k = recv_offsets[i]; k < recv_offsets[i + 1]; ++k)
        {
          if (std::int32_t c = recv_cols_bucketed[k]; last_seen[c] != i)
          {
            last_seen[c] = i;
            _edges.push_back(c);
          }
        }
      }
      else
      {
        // Nothing was received for this row, so the de-duplicated cache is
        // the row. Skipping the marker pass leaves last_seen untouched,
        // which is safe: rows are visited in increasing order and the row
        // index is the stamp, so a stamp left by an earlier row never
        // matches a later one.
        _edges.insert(_edges.end(),
                      std::next(cache_cols.begin(), cache_offsets[i]),
                      std::next(cache_cols.begin(), cache_offsets[i + 1]));
      }

      std::vector<std::int32_t>::iterator row_start
          = std::next(_edges.begin(), row_begin);
      std::ranges::sort(row_start, _edges.end());

      // Find position of first "off-diagonal" column
      _off_diagonal_offsets[i] = std::ranges::distance(
          row_start,
          std::ranges::lower_bound(row_start, _edges.end(), local_size1));

      _offsets.push_back(static_cast<std::int64_t>(_edges.size()));
    }
  }

  // _col_ghosts only appends to the original column ghosts. Rebuild the
  // collective IndexMap only if ghosts changed on at least one rank;
  // otherwise preserve a shared row and column IndexMap.
  int ghosts_changed = _col_ghosts.size() != _index_maps[1]->ghosts().size();
  int ghosts_changed_global;
  const int ierr = MPI_Allreduce(&ghosts_changed, &ghosts_changed_global, 1,
                                 MPI_INT, MPI_LOR, _comm.comm());
  dolfinx::MPI::check_error(_comm.comm(), ierr);
  if (ghosts_changed_global)
  {
    spdlog::debug("Column ghost size increased from {} to {}",
                  _index_maps[1]->ghosts().size(), _col_ghosts.size());
    _index_maps[1] = std::make_shared<common::IndexMap>(
        _comm.comm(), _index_maps[1]->size_local(), _col_ghosts,
        _col_ghost_owners);
  }
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
