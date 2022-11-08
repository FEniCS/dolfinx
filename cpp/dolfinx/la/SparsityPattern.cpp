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
#include <dolfinx/graph/AdjacencyList.h>
#include <map>

using namespace dolfinx;
using namespace dolfinx::la;

//-----------------------------------------------------------------------------
SparsityPattern::SparsityPattern(
    MPI_Comm comm,
    const std::array<std::shared_ptr<const common::IndexMap>, 2>& maps,
    const std::array<int, 2>& bs)
    : _comm(comm), _index_maps(maps), _bs(bs),
      _row_cache(maps[0]->size_local() + maps[0]->num_ghosts())
{
  assert(maps[0]);
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

  _row_cache.resize(_index_maps[0]->size_local()
                    + _index_maps[0]->num_ghosts());
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

      if (p->_graph)
      {
        throw std::runtime_error("Sub-sparsity pattern has been finalised. "
                                 "Cannot compute stacked pattern.");
      }

      const int bs_dof0 = bs[0][row];
      const int bs_dof1 = bs[1][col];

      // Iterate over owned rows cache
      for (std::int32_t i = 0; i < num_rows_local; ++i)
      {
        for (std::int32_t c_old : p->_row_cache[i])
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
              _row_cache[r_new + k0].push_back(c_new + k1);
          }
        }
      }
      // Iterate over unowned rows cache
      for (std::int32_t i = 0; i < num_ghost_rows_local; ++i)
      {
        for (std::int32_t c_old : p->_row_cache[i + num_rows_local])
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
              _row_cache[num_rows_local_new + r_new + k0].push_back(c_new + k1);
          }
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert(const std::span<const std::int32_t>& rows,
                             const std::span<const std::int32_t>& cols)
{
  if (_graph)
  {
    throw std::runtime_error(
        "Cannot insert into sparsity pattern. It has already been assembled");
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

    _row_cache[row].insert(_row_cache[row].end(), cols.begin(), cols.end());
  }
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert_diagonal(std::span<const std::int32_t> rows)
{
  if (_graph)
  {
    throw std::runtime_error(
        "Cannot insert into sparsity pattern. It has already been assembled");
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

    _row_cache[row].push_back(row);
  }
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
  if (!_graph)
    throw std::runtime_error("Sparsity pattern has not been finalised.");

  std::array range = _index_maps[1]->local_range();
  const std::int32_t local_size = range[1] - range[0];
  const std::int32_t num_ghosts = _col_ghosts.size();
  std::vector<std::int64_t> global(local_size + num_ghosts);
  std::iota(global.begin(), std::next(global.begin(), local_size), range[0]);
  std::copy(_col_ghosts.begin(), _col_ghosts.end(),
            global.begin() + local_size);
  return global;
}
//-----------------------------------------------------------------------------
common::IndexMap SparsityPattern::column_index_map() const
{
  if (!_graph)
    throw std::runtime_error("Sparsity pattern has not been finalised.");

  std::array range = _index_maps[1]->local_range();
  const std::int32_t local_size = range[1] - range[0];
  return common::IndexMap(_comm.comm(), local_size, _col_ghosts,
                          _col_ghost_owners);
}
//-----------------------------------------------------------------------------
int SparsityPattern::block_size(int dim) const { return _bs[dim]; }
//-----------------------------------------------------------------------------
void SparsityPattern::assemble()
{
  if (_graph)
    throw std::runtime_error("Sparsity pattern has already been finalised.");

  common::Timer t0("SparsityPattern::assemble");

  assert(_index_maps[0]);
  const std::int32_t local_size0 = _index_maps[0]->size_local();
  const std::array local_range0 = _index_maps[0]->local_range();
  const std::vector<std::int64_t>& ghosts0 = _index_maps[0]->ghosts();

  const std::vector<int>& owners0 = _index_maps[0]->owners();
  const std::vector<int>& src0 = _index_maps[0]->src();

  assert(_index_maps[1]);
  const std::int32_t local_size1 = _index_maps[1]->size_local();
  const std::array local_range1 = _index_maps[1]->local_range();

  _col_ghosts = _index_maps[1]->ghosts();
  _col_ghost_owners = _index_maps[1]->owners();

  // Compute size of data to send to each process
  std::vector<int> send_sizes(src0.size(), 0);
  for (std::size_t i = 0; i < owners0.size(); ++i)
  {
    auto it = std::lower_bound(src0.begin(), src0.end(), owners0[i]);
    assert(it != src0.end() and *it == owners0[i]);
    const int neighbour_rank = std::distance(src0.begin(), it);
    send_sizes[neighbour_rank] += 3 * _row_cache[i + local_size0].size();
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
    auto it = std::lower_bound(src0.begin(), src0.end(), owners0[i]);
    assert(it != src0.end() and *it == owners0[i]);
    const int neighbour_rank = std::distance(src0.begin(), it);

    for (std::int32_t col_local : _row_cache[i + local_size0])
    {
      // Get index in send buffer
      const std::int32_t pos = insert_pos[neighbour_rank];

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

      insert_pos[neighbour_rank] += 3;
    }
  }

  // Exchange data between processes
  std::vector<std::int64_t> ghost_data_in;
  {
    MPI_Comm comm;
    const std::vector<int>& dest0 = _index_maps[0]->dest();
    MPI_Dist_graph_create_adjacent(
        _index_maps[0]->comm(), dest0.size(), dest0.data(), MPI_UNWEIGHTED,
        src0.size(), src0.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm);

    std::vector<int> recv_sizes(dest0.size());
    send_sizes.reserve(1);
    recv_sizes.reserve(1);
    MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1,
                          MPI_INT, comm);

    // Build recv displacements
    std::vector<int> recv_disp = {0};
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
  std::map<std::int64_t, std::int32_t> global_to_local;
  std::int32_t local_i = local_size1;
  for (std::int64_t global_i : _col_ghosts)
    global_to_local.insert({global_i, local_i++});

  // Add data received from the neighborhood
  for (std::size_t i = 0; i < ghost_data_in.size(); i += 3)
  {
    const std::int32_t row_local = ghost_data_in[i] - local_range0[0];
    const std::int64_t col = ghost_data_in[i + 1];
    const int owner = ghost_data_in[i + 2];
    if (col >= local_range1[0] and col < local_range1[1])
    {
      // Convert to local column index
      const std::int32_t J = col - local_range1[0];
      _row_cache[row_local].push_back(J);
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

      const std::int32_t col_local = it.first->second;
      _row_cache[row_local].push_back(col_local);
    }
  }

  // Sort and remove duplicate column indices in each row
  std::vector<std::int32_t> adj_counts(local_size0 + owners0.size(), 0);
  std::vector<std::int32_t> adj_data;
  _off_diagonal_offset.resize(local_size0 + owners0.size());
  for (std::size_t i = 0; i < local_size0 + owners0.size(); ++i)
  {
    std::vector<std::int32_t>& row = _row_cache[i];
    std::sort(row.begin(), row.end());
    auto it_end = std::unique(row.begin(), row.end());

    // Find position of first "off-diagonal" column
    _off_diagonal_offset[i] = std::distance(
        row.begin(), std::lower_bound(row.begin(), it_end, local_size1));

    adj_data.insert(adj_data.end(), row.begin(), it_end);
    adj_counts[i] += std::distance(row.begin(), it_end);
  }
  // Clear cache
  std::vector<std::vector<std::int32_t>>().swap(_row_cache);

  // Compute offsets for adjacency list
  std::vector<std::int32_t> adj_offsets(local_size0 + owners0.size() + 1);
  std::partial_sum(adj_counts.begin(), adj_counts.end(),
                   adj_offsets.begin() + 1);

  adj_data.shrink_to_fit();
  _graph = std::make_shared<graph::AdjacencyList<std::int32_t>>(
      std::move(adj_data), std::move(adj_offsets));

  // Column count increased due to received rows from other processes
  LOG(INFO) << "Column ghost size increased from "
            << _index_maps[1]->ghosts().size() << " to " << _col_ghosts.size()
            << std::endl;
}
//-----------------------------------------------------------------------------
std::int64_t SparsityPattern::num_nonzeros() const
{
  if (!_graph)
    throw std::runtime_error("Sparsity pattern has not be assembled.");
  return _graph->array().size();
}
//-----------------------------------------------------------------------------
std::int32_t SparsityPattern::nnz_diag(std::int32_t row) const
{
  if (!_graph)
    throw std::runtime_error("Sparsity pattern has not be assembled.");
  return _off_diagonal_offset[row];
}
//-----------------------------------------------------------------------------
std::int32_t SparsityPattern::nnz_off_diag(std::int32_t row) const
{
  if (!_graph)
    throw std::runtime_error("Sparsity pattern has not be assembled.");
  return _graph->num_links(row) - _off_diagonal_offset[row];
}
//-----------------------------------------------------------------------------
const graph::AdjacencyList<std::int32_t>& SparsityPattern::graph() const
{
  if (!_graph)
    throw std::runtime_error("Sparsity pattern has not been assembled.");
  return *_graph;
}
//-----------------------------------------------------------------------------
std::span<const int> SparsityPattern::off_diagonal_offset() const
{
  return _off_diagonal_offset;
}
//-----------------------------------------------------------------------------
MPI_Comm SparsityPattern::comm() const { return _comm.comm(); }
//-----------------------------------------------------------------------------
