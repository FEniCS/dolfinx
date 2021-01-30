// Copyright (C) 2007-2020 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "SparsityPattern.h"
#include <algorithm>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/graph/AdjacencyList.h>

using namespace dolfinx;
using namespace dolfinx::la;

//-----------------------------------------------------------------------------
SparsityPattern::SparsityPattern(
    MPI_Comm comm,
    const std::array<std::shared_ptr<const common::IndexMap>, 2>& maps,
    const std::array<int, 2>& bs)
    : _mpi_comm(comm), _index_maps(maps), _bs(bs)
{
  assert(maps[0]);
  _cache_owned.resize(maps[0]->size_local());
  _cache_unowned.resize(maps[0]->num_ghosts());
}
//-----------------------------------------------------------------------------
SparsityPattern::SparsityPattern(
    MPI_Comm comm,
    const std::vector<std::vector<const SparsityPattern*>>& patterns,
    const std::array<std::vector<std::pair<
                         std::reference_wrapper<const common::IndexMap>, int>>,
                     2>& maps,
    const std::array<std::vector<int>, 2>& bs)
    : _mpi_comm(comm), _bs({1, 1})
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
      comm, local_offset0.back(),
      dolfinx::MPI::compute_graph_edges(
          comm, std::set<int>(ghost_owners0.begin(), ghost_owners0.end())),
      ghosts0, ghost_owners0);
  _index_maps[1] = std::make_shared<common::IndexMap>(
      comm, local_offset1.back(),
      dolfinx::MPI::compute_graph_edges(
          comm, std::set<int>(ghost_owners1.begin(), ghost_owners1.end())),
      ghosts1, ghost_owners1);

  _cache_owned.resize(_index_maps[0]->size_local());
  _cache_unowned.resize(_index_maps[0]->num_ghosts());

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

      if (p->_diagonal)
      {
        throw std::runtime_error("Sub-sparsity pattern has been finalised. "
                                 "Cannot compute stacked pattern.");
      }

      const int bs_dof0 = bs[0][row];
      const int bs_dof1 = bs[1][col];

      // Iterate over owned rows cache
      for (std::int32_t i = 0; i < num_rows_local; ++i)
      {
        for (std::int32_t c_old : p->_cache_owned[i])
        {
          const std::int32_t r_new = bs_dof0 * i + local_offset0[row];
          const std::int32_t c_new = (c_old < num_cols_local)
                                         ? bs_dof1 * c_old + local_offset1[col]
                                         : bs_dof1 * (c_old - num_cols_local)
                                               + local_offset1.back()
                                               + ghost_offsets1[col];

          for (int k0 = 0; k0 < bs_dof0; ++k0)
          {
            for (int k1 = 0; k1 < bs_dof1; ++k1)
              _cache_owned[r_new + k0].push_back(c_new + k1);
          }
        }
      }
      // Iterate over unowned rows cache
      for (std::int32_t i = 0; i < num_ghost_rows_local; ++i)
      {
        for (std::int32_t c_old : p->_cache_unowned[i])
        {
          const std::int32_t r_new = bs_dof0 * i + ghost_offsets0[row];
          const std::int32_t c_new = (c_old < num_cols_local)
                                         ? bs_dof1 * c_old + local_offset1[col]
                                         : bs_dof1 * (c_old - num_cols_local)
                                               + local_offset1.back()
                                               + ghost_offsets1[col];
          for (int k0 = 0; k0 < bs_dof0; ++k0)
          {
            for (int k1 = 0; k1 < bs_dof1; ++k1)
              _cache_unowned[r_new + k0].push_back(c_new + k1);
          }
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
std::shared_ptr<const common::IndexMap>
SparsityPattern::index_map(int dim) const
{
  return _index_maps.at(dim);
}
//-----------------------------------------------------------------------------
int SparsityPattern::block_size(int dim) const { return _bs[dim]; }
//-----------------------------------------------------------------------------
void SparsityPattern::insert(const tcb::span<const std::int32_t>& rows,
                             const tcb::span<const std::int32_t>& cols)
{
  if (_diagonal)
  {
    throw std::runtime_error(
        "Cannot insert into sparsity pattern. It has already been assembled");
  }

  assert(_index_maps[0]);
  const std::int32_t local_size0 = _index_maps[0]->size_local();
  const std::int32_t size0 = local_size0 + _index_maps[0]->num_ghosts();

  for (std::int32_t row : rows)
  {
    if (row < local_size0)
      _cache_owned[row].insert(_cache_owned[row].end(), cols.begin(),
                               cols.end());
    else if (row < size0)
    {
      _cache_unowned[row - local_size0].insert(
          _cache_owned[row - local_size0].end(), cols.begin(), cols.end());
    }
    else
    {
      throw std::runtime_error(
          "Cannot insert rows that do not exist in the IndexMap.");
    }
  }
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert_diagonal(const std::vector<int32_t>& rows)
{
  if (_diagonal)
  {
    throw std::runtime_error(
        "Cannot insert into sparsity pattern. It has already been assembled");
  }

  assert(_index_maps[0]);
  const std::int32_t local_size0 = _index_maps[0]->size_local();
  const std::int32_t size0 = local_size0 + _index_maps[0]->num_ghosts();

  for (std::int32_t row : rows)
  {
    if (row < local_size0)
      _cache_owned[row].push_back(row);
    else if (row < size0)
      _cache_unowned[row - local_size0].push_back(row);
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

  common::Timer t0("SparsityPattern::assemble");

  assert(_index_maps[0]);
  const std::int32_t local_size0 = _index_maps[0]->size_local();
  const std::int32_t num_ghosts0 = _index_maps[0]->num_ghosts();
  const std::array local_range0 = _index_maps[0]->local_range();
  const std::vector<std::int64_t>& ghosts0 = _index_maps[0]->ghosts();

  assert(_index_maps[1]);
  const std::int32_t local_size1 = _index_maps[1]->size_local();
  const std::array local_range1 = _index_maps[1]->local_range();
  std::vector<std::int64_t> ghosts1 = _index_maps[1]->ghosts();
  std::vector<int> ghost_owners1 = _index_maps[1]->ghost_owner_rank();

  // Global to local map for ghost columns
  const int mpi_rank = dolfinx::MPI::rank(_mpi_comm.comm());
  std::map<std::int64_t, std::int32_t> global_to_local;
  std::int32_t local_i = local_size1;
  for (std::int64_t global_i : ghosts1)
    global_to_local.insert({global_i, local_i++});

  // For each ghost row, pack and send (global row, global col) pairs to send to
  // neighborhood
  std::vector<std::int64_t> ghost_data;
  for (int i = 0; i < num_ghosts0; ++i)
  {
    const std::int64_t row_global = ghosts0[i];
    for (std::int32_t col_local : _cache_unowned[i])
    {
      ghost_data.push_back(row_global);
      if (col_local < local_size1)
      {
        ghost_data.push_back(col_local + local_range1[0]);
        ghost_data.push_back(mpi_rank);
      }
      else
      {
        ghost_data.push_back(ghosts1[col_local - local_size1]);
        ghost_data.push_back(ghost_owners1[col_local - local_size1]);
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
  for (std::size_t i = 0; i < ghost_data_received.size(); i += 3)
  {
    const std::int64_t row = ghost_data_received[i];
    if (row >= local_range0[0] and row < local_range0[1])
    {
      const std::int32_t row_local = row - local_range0[0];
      const std::int64_t col = ghost_data_received[i + 1];
      if (col >= local_range1[0] and col < local_range1[1])
      {
        // Convert to local column index
        const std::int32_t J = col - local_range1[0];
        _cache_owned[row_local].push_back(J);
      }
      else
      {
        // Column index may not exist in column indexmap
        auto it = global_to_local.insert({col, local_i});
        if (it.second)
        {
          ghosts1.push_back(col);
          ghost_owners1.push_back(ghost_data_received[i + 2]);
          ++local_i;
        }
        const std::int32_t col_local = it.first->second;
        _cache_owned[row_local].push_back(col_local);
      }
    }
  }

  // Sort and remove duplicates
  std::vector<std::int32_t> adj_counts(local_size0, 0);
  std::vector<std::int32_t> adj_data;
  std::vector<std::int32_t> adj_counts_off(local_size0, 0);
  std::vector<std::int32_t> adj_data_off;
  for (std::int32_t i = 0; i < local_size0; ++i)
  {
    std::vector<std::int32_t>& row = _cache_owned[i];
    std::sort(row.begin(), row.end());
    const std::vector<std::int32_t>::iterator it_end
        = std::unique(row.begin(), row.end());

    // Find position of first "off-diagonal" column
    const std::vector<std::int32_t>::iterator it_diag
        = std::lower_bound(row.begin(), it_end, local_size1);

    adj_data.insert(adj_data.end(), row.begin(), it_diag);
    adj_counts[i] += (it_diag - row.begin());
    adj_data_off.insert(adj_data_off.end(), it_diag, it_end);
    adj_counts_off[i] += (it_end - it_diag);
    std::vector<std::int32_t>().swap(row);
  }
  std::vector<std::vector<std::int32_t>>().swap(_cache_owned);

  std::vector<std::int32_t> adj_offsets(local_size0 + 1);
  std::partial_sum(adj_counts.begin(), adj_counts.end(),
                   adj_offsets.begin() + 1);
  std::vector<std::int32_t> adj_offsets_off(local_size0 + 1);
  std::partial_sum(adj_counts_off.begin(), adj_counts_off.end(),
                   adj_offsets_off.begin() + 1);

  // FIXME: after assembly, there are no ghost rows, i.e. the IndexMap for rows
  // should be non-overlapping. However, we are retaining the row overlap
  // information and associated mapping, as this will be needed for matrix
  // assembly.
  //
  // _index_maps[0] = std::make_shared<common::IndexMap>(comm, local_size0);
  _diagonal = std::make_shared<graph::AdjacencyList<std::int32_t>>(
      std::move(adj_data), std::move(adj_offsets));

  // Column map increased due to received rows from other processes (see above)
  LOG(INFO) << "Column ghost size increased from "
            << _index_maps[1]->ghosts().size() << " to " << ghosts1.size()
            << "\n";

  _index_maps[1] = std::make_shared<common::IndexMap>(
      _mpi_comm.comm(), local_size1,
      dolfinx::MPI::compute_graph_edges(
          _mpi_comm.comm(),
          std::set<int>(ghost_owners1.begin(), ghost_owners1.end())),
      ghosts1, ghost_owners1);

  _off_diagonal = std::make_shared<graph::AdjacencyList<std::int32_t>>(
      std::move(adj_data_off), std::move(adj_offsets_off));
}
//-----------------------------------------------------------------------------
std::int64_t SparsityPattern::num_nonzeros() const
{
  if (!_diagonal)
    throw std::runtime_error("Sparsity pattern has not be assembled.");
  assert(_off_diagonal);
  return _diagonal->array().size() + _off_diagonal->array().size();
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
const graph::AdjacencyList<std::int32_t>&
SparsityPattern::off_diagonal_pattern() const
{
  if (!_off_diagonal)
    throw std::runtime_error("Sparsity pattern has not been finalised.");
  return *_off_diagonal;
}
//-----------------------------------------------------------------------------
MPI_Comm SparsityPattern::mpi_comm() const { return _mpi_comm.comm(); }
//-----------------------------------------------------------------------------
