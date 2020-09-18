// Copyright (C) 2007-2020 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "SparsityPattern.h"
#include <algorithm>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/log.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/graph/AdjacencyList.h>

using namespace dolfinx;
using namespace dolfinx::la;

//-----------------------------------------------------------------------------
SparsityPattern::SparsityPattern(
    MPI_Comm comm,
    const std::array<std::shared_ptr<const common::IndexMap>, 2>& index_maps)
    : _mpi_comm(comm), _index_maps(index_maps)
{
  const std::int32_t local_size0
      = index_maps[0]->block_size()
        * (index_maps[0]->size_local() + index_maps[0]->num_ghosts());
  _diagonal_cache.resize(local_size0);
  _off_diagonal_cache.resize(local_size0);
}
//-----------------------------------------------------------------------------
SparsityPattern::SparsityPattern(
    MPI_Comm comm,
    const std::vector<std::vector<const SparsityPattern*>>& patterns,
    const std::array<
        std::vector<std::reference_wrapper<const common::IndexMap>>, 2>& maps)
    : _mpi_comm(comm)
{
  // FIXME: - Add range/bound checks for each block
  //        - Check for compatible block sizes for each block

  const auto [rank_offset0, local_offset0, ghosts_new0, owners0]
      = common::stack_index_maps(maps[0]);
  const auto [rank_offset1, local_offset1, ghosts_new1, owners1]
      = common::stack_index_maps(maps[1]);

  std::vector<std::int64_t> ghosts0, ghosts1;
  std::vector<std::int32_t> ghost_offsets0(1, 0);
  for (const std::vector<std::int64_t>& ghosts : ghosts_new0)
  {
    ghost_offsets0.push_back(ghost_offsets0.back() + ghosts.size());
    ghosts0.insert(ghosts0.end(), ghosts.begin(), ghosts.end());
  }
  for (const std::vector<std::int64_t>& ghosts : ghosts_new1)
    ghosts1.insert(ghosts1.end(), ghosts.begin(), ghosts.end());

  std::vector<int> ghost_owners0, ghost_owners1;
  for (const std::vector<int>& owners : owners0)
    ghost_owners0.insert(ghost_owners0.end(), owners.begin(), owners.end());
  for (const std::vector<int>& owners : owners1)
    ghost_owners1.insert(ghost_owners1.end(), owners.begin(), owners.end());

  // Create new IndexMaps
  _index_maps[0] = std::make_shared<common::IndexMap>(
      comm, local_offset0.back(),
      dolfinx::MPI::compute_graph_edges(
          comm, std::set<int>(ghost_owners0.begin(), ghost_owners0.end())),
      ghosts0, ghost_owners0, 1);
  _index_maps[1] = std::make_shared<common::IndexMap>(
      comm, local_offset1.back(),
      dolfinx::MPI::compute_graph_edges(
          comm, std::set<int>(ghost_owners1.begin(), ghost_owners1.end())),
      ghosts1, ghost_owners1, 1);

  // Size cache arrays
  const std::int32_t size_row = local_offset0.back() + ghosts0.size();
  _diagonal_cache.resize(size_row);
  _off_diagonal_cache.resize(size_row);

  // NOTE: This could be simplified if we used local column indices
  // Build map from old column global indices to new column global
  // indices
  std::vector<std::map<std::int64_t, std::int64_t>> col_old_to_new(
      maps[1].size());
  for (std::size_t col = 0; col < maps[1].size(); ++col)
  {
    auto map = maps[1][col];
    const int bs = map.get().block_size();
    const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& ghosts_old
        = map.get().ghosts();
    assert(ghosts_new1[col].size() == (std::size_t)(bs * ghosts_old.rows()));
    for (int i = 0; i < ghosts_old.rows(); ++i)
    {
      for (int j = 0; j < bs; ++j)
      {
        col_old_to_new[col].insert(
            {bs * ghosts_old[i] + j, ghosts_new1[col][i * bs + j]});
      }
    }
  }

  // Iterate over block rows
  for (std::size_t row = 0; row < patterns.size(); ++row)
  {
    const common::IndexMap& map_row = maps[0][row].get();
    const std::int32_t num_rows_local
        = map_row.size_local() * map_row.block_size();
    const std::int32_t num_rows_ghost
        = map_row.num_ghosts() * map_row.block_size();

    // Iterate over block columns of current row (block)
    for (std::size_t col = 0; col < patterns[row].size(); ++col)
    {
      // Get pattern for this block
      const SparsityPattern* p = patterns[row][col];
      if (!p)
        continue;

      if (p->_diagonal)
      {
        throw std::runtime_error("Sub-sparsity pattern has been finalised. "
                                 "Cannot compute stacked pattern.");
      }

      // Loop over owned rows
      for (int i = 0; i < num_rows_local; ++i)
      {
        // New local row index
        const std::int32_t r_new = i + local_offset0[row];

        // Insert diagonal block entries (local column indices)
        const std::vector<std::int32_t>& cols = p->_diagonal_cache[i];
        for (std::size_t j = 0; j < cols.size(); ++j)
          _diagonal_cache[r_new].push_back(cols[j] + local_offset1[col]);

        // Insert Off-diagonal block entries (global column indices)
        const std::vector<std::int64_t>& cols_off = p->_off_diagonal_cache[i];
        for (std::size_t j = 0; j < cols_off.size(); ++j)
        {
          auto it = col_old_to_new[col].find(cols_off[j]);
          assert(it != col_old_to_new[col].end());
          _off_diagonal_cache[r_new].push_back(it->second);
        }
      }

      // Loop over ghost rows
      for (int i = 0; i < num_rows_ghost; ++i)
      {
        // New local row index
        const std::int32_t r_new
            = i + local_offset0.back() + ghost_offsets0[row];

        // Insert diagonal block entries (local column indices)
        const std::vector<std::int32_t>& cols
            = p->_diagonal_cache[num_rows_local + i];
        for (std::size_t j = 0; j < cols.size(); ++j)
          _diagonal_cache[r_new].push_back(cols[j] + local_offset1[col]);

        // Off-diagonal block entries (global column indices)
        const std::vector<std::int64_t>& cols_off
            = p->_off_diagonal_cache[num_rows_local + i];
        for (std::size_t j = 0; j < cols_off.size(); ++j)
        {
          // Get local index and convert to global (for this block)
          auto it = col_old_to_new[col].find(cols_off[j]);
          _off_diagonal_cache[r_new].push_back(it->second);
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> SparsityPattern::local_range(int dim) const
{
  const int bs = _index_maps.at(dim)->block_size();
  const std::array lrange = _index_maps[dim]->local_range();
  return {bs * lrange[0], bs * lrange[1]};
}
//-----------------------------------------------------------------------------
std::shared_ptr<const common::IndexMap>
SparsityPattern::index_map(int dim) const
{
  return _index_maps.at(dim);
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert(
    const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>& rows,
    const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>& cols)
{
  if (_diagonal)
  {
    throw std::runtime_error(
        "Cannot insert into sparsity pattern. It has already been assembled");
  }

  assert(_index_maps[0]);
  const int bs0 = _index_maps[0]->block_size();
  const std::int32_t size0
      = bs0 * (_index_maps[0]->size_local() + _index_maps[0]->num_ghosts());

  assert(_index_maps[1]);
  const int bs1 = _index_maps[1]->block_size();
  const std::int32_t local_size1 = _index_maps[1]->size_local();
  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& ghosts1
      = _index_maps[1]->ghosts();

  for (Eigen::Index i = 0; i < rows.rows(); ++i)
  {
    if (rows[i] < size0)
    {
      for (Eigen::Index j = 0; j < cols.rows(); ++j)
      {
        if (cols[j] < bs1 * local_size1)
          _diagonal_cache[rows[i]].push_back(cols[j]);
        else
        {
          const std::div_t div = std::div(cols[j], bs1);
          const std::int64_t block_global = ghosts1[div.quot - local_size1];
          _off_diagonal_cache[rows[i]].push_back(bs1 * block_global + div.rem);
        }
      }
    }
    else
    {
      throw std::runtime_error(
          "Cannot insert rows that do not exist in the IndexMap.");
    }
  }
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert_diagonal(
    const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>& rows)
{
  if (_diagonal)
  {
    throw std::runtime_error(
        "Cannot insert into sparsity pattern. It has already been assembled");
  }

  assert(_index_maps[0]);
  const int bs0 = _index_maps[0]->block_size();
  const std::int32_t local_size0
      = bs0 * (_index_maps[0]->size_local() + _index_maps[0]->num_ghosts());
  for (Eigen::Index i = 0; i < rows.rows(); ++i)
  {
    if (rows[i] < local_size0)
      _diagonal_cache[rows[i]].push_back(rows[i]);
    else
    {
      throw std::runtime_error(
          "Cannot insert rows that do not exist in the IndexMap.");
    }
  }
}
//-----------------------------------------------------------------------------
void SparsityPattern::assemble()
{
  if (_diagonal)
    throw std::runtime_error("Sparsity pattern has already been finalised.");
  assert(!_off_diagonal);

  assert(_index_maps[0]);
  const int bs0 = _index_maps[0]->block_size();
  const std::int32_t local_size0 = _index_maps[0]->size_local();
  const std::int32_t num_ghosts0 = _index_maps[0]->num_ghosts();
  const std::array local_range0 = _index_maps[0]->local_range();
  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& ghosts0
      = _index_maps[0]->ghosts();

  assert(_index_maps[1]);
  const int bs1 = _index_maps[1]->block_size();
  const std::int32_t local_size1 = _index_maps[1]->size_local();
  const std::array local_range1 = _index_maps[1]->local_range();
  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& ghosts1
      = _index_maps[1]->ghosts();

  // For each ghost row, pack and send (global row, global col) pairs to
  // send to neighborhood
  std::vector<std::int64_t> ghost_data;
  for (int i = 0; i < num_ghosts0; ++i)
  {
    const std::int64_t row_node_global = ghosts0[i];
    const std::int32_t row_node_local = local_size0 + i;
    for (int j = 0; j < bs0; ++j)
    {
      const std::int64_t row_global = bs0 * row_node_global + j;
      const std::int32_t row_local = bs0 * row_node_local + j;
      assert((std::size_t)row_local < _diagonal_cache.size());
      const std::vector<std::int32_t>& cols = _diagonal_cache[row_local];
      for (std::size_t c = 0; c < cols.size(); ++c)
      {
        ghost_data.push_back(row_global);

        // Convert to global column index
        if (cols[c] < bs1 * local_size1)
          ghost_data.push_back(cols[c] + bs1 * local_range1[0]);
        else
        {
          const std::div_t div = std::div(cols[c], bs1);
          const std::int64_t block_global = ghosts1[div.quot - local_size1];
          ghost_data.push_back(bs1 * block_global + div.rem);
        }
      }

      const std::vector<std::int64_t>& cols_off
          = _off_diagonal_cache[row_local];
      for (std::size_t c = 0; c < cols_off.size(); ++c)
      {
        ghost_data.push_back(row_global);
        ghost_data.push_back(cols_off[c]);
      }
    }
  }

  MPI_Comm comm = _index_maps[0]->comm(common::IndexMap::Direction::symmetric);
  int num_neighbors(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(comm, &num_neighbors, &outdegree, &weighted);
  assert(num_neighbors == outdegree);

  // Figure out how much data to receive from each neighbor
  const int num_my_rows = ghost_data.size();
  std::vector<int> num_rows_recv(num_neighbors);
  MPI_Neighbor_allgather(&num_my_rows, 1, MPI_INT, num_rows_recv.data(), 1,
                         MPI_INT, comm);

  // Compute displacements for data to receive
  std::vector<int> disp(num_neighbors + 1, 0);
  std::partial_sum(num_rows_recv.begin(), num_rows_recv.end(),
                   disp.begin() + 1);

  // NOTE: Send unowned rows to all neighbors could be a bit 'lazy' and
  // MPI_Neighbor_alltoallv could be used to send just to the owner, but
  // maybe the number of rows exchanged in the neighborhood are
  // relatively small that MPI_Neighbor_allgatherv is simpler.

  // Send all unowned rows to neighbors, and receive rows from
  // neighbors
  std::vector<std::int64_t> ghost_data_received(disp.back());
  MPI_Neighbor_allgatherv(ghost_data.data(), ghost_data.size(), MPI_INT64_T,
                          ghost_data_received.data(), num_rows_recv.data(),
                          disp.data(), MPI_INT64_T, comm);

  // Add data received from the neighborhood
  for (std::size_t i = 0; i < ghost_data_received.size(); i += 2)
  {
    const std::int64_t row = ghost_data_received[i];
    if (row >= bs0 * local_range0[0] and row < bs0 * local_range0[1])
    {
      const std::int32_t row_local = row - bs0 * local_range0[0];
      const std::int64_t col = ghost_data_received[i + 1];
      if (col >= bs1 * local_range1[0] and col < bs1 * local_range1[1])
      {
        // Convert to local column index
        const std::int32_t J = col - bs1 * local_range1[0];
        _diagonal_cache[row_local].push_back(J);
      }
      else
      {
        assert(row_local < (std::int32_t)_off_diagonal_cache.size());
        _off_diagonal_cache[row_local].push_back(col);
      }
    }
  }

  _diagonal_cache.resize(bs0 * local_size0);
  for (std::vector<std::int32_t>& row : _diagonal_cache)
  {
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
  }
  _diagonal
      = std::make_shared<graph::AdjacencyList<std::int32_t>>(_diagonal_cache);
  std::vector<std::vector<std::int32_t>>().swap(_diagonal_cache);

  _off_diagonal_cache.resize(bs0 * local_size0);
  for (std::vector<std::int64_t>& row : _off_diagonal_cache)
  {
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
  }
  _off_diagonal = std::make_shared<graph::AdjacencyList<std::int64_t>>(
      _off_diagonal_cache);
  std::vector<std::vector<std::int64_t>>().swap(_off_diagonal_cache);
}
//-----------------------------------------------------------------------------
std::int64_t SparsityPattern::num_nonzeros() const
{
  if (!_diagonal)
    throw std::runtime_error("Sparsity pattern has not be assembled.");
  assert(_off_diagonal);
  return _diagonal->array().rows() + _off_diagonal->array().rows();
}
//-----------------------------------------------------------------------------
const graph::AdjacencyList<std::int32_t>&
SparsityPattern::diagonal_pattern() const
{
  if (!_diagonal)
    throw std::runtime_error("Sparsity pattern has not been finalised.");
  return *_diagonal;
}
//-----------------------------------------------------------------------------
const graph::AdjacencyList<std::int64_t>&
SparsityPattern::off_diagonal_pattern() const
{
  if (!_off_diagonal)
    throw std::runtime_error("Sparsity pattern has not been finalised.");
  return *_off_diagonal;
}
//-----------------------------------------------------------------------------
MPI_Comm SparsityPattern::mpi_comm() const { return _mpi_comm.comm(); }
//-----------------------------------------------------------------------------
