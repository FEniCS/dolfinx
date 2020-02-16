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

namespace
{
const auto col_map = [](const std::int32_t j_index,
                        const common::IndexMap& index_map1) -> std::int64_t {
  const int bs = index_map1.block_size;
  const std::div_t div = std::div(j_index, bs);
  const int component = div.rem;
  const int index = div.quot;
  return bs * index_map1.local_to_global(index) + component;
};

} // namespace

//-----------------------------------------------------------------------------
SparsityPattern::SparsityPattern(
    MPI_Comm comm,
    const std::array<std::shared_ptr<const common::IndexMap>, 2>& index_maps)
    : _mpi_comm(comm), _index_maps(index_maps)
{
  const std::int32_t local_size0
      = index_maps[0]->block_size
        * (index_maps[0]->size_local() + index_maps[0]->num_ghosts());
  _diagonal_cache.resize(local_size0);
  _off_diagonal_cache.resize(local_size0);
}
//-----------------------------------------------------------------------------
SparsityPattern::SparsityPattern(
    MPI_Comm comm,
    const std::vector<std::vector<const SparsityPattern*>>& patterns)
    : _mpi_comm(comm)
{
  // FIXME: - Add range/bound checks for each block
  //        - Check for compatible block sizes for each block
  //        - Support null blocks (maybe insist on null block having
  //          common::IndexMaps?)

  std::vector<std::vector<std::int32_t>> diagonal;
  std::vector<std::vector<std::int64_t>> off_diagonal;

  // Get row ranges using column 0
  std::int64_t row_global_offset(0), row_local_size(0);
  for (std::size_t row = 0; row < patterns.size(); ++row)
  {
    assert(patterns[row][0]);
    assert(patterns[row][0]->_index_maps[0]);
    auto local_range = patterns[row][0]->_index_maps[0]->local_range();
    const int bs0 = patterns[row][0]->_index_maps[0]->block_size;
    row_global_offset += bs0 * local_range[0];
    row_local_size += bs0 * (local_range[1] - local_range[0]);
  }

  // Get column ranges using row 0
  std::int64_t col_process_offset(0), col_local_size(0);
  std::vector<const common::IndexMap*> cmaps;
  for (std::size_t col = 0; col < patterns[0].size(); ++col)
  {
    assert(patterns[0][col]);
    assert(patterns[0][col]->_index_maps[1]);
    cmaps.push_back(patterns[0][col]->_index_maps[1].get());
    auto local_range = patterns[0][col]->_index_maps[1]->local_range();
    const int bs1 = patterns[0][col]->_index_maps[1]->block_size;
    col_process_offset += bs1 * local_range[0];
    col_local_size += bs1 * (local_range[1] - local_range[0]);
  }

  // Iterate over block rows
  std::int64_t row_local_offset = 0;
  for (std::size_t row = 0; row < patterns.size(); ++row)
  {
    // Increase storage for nodes
    assert(patterns[row][0]);
    assert(patterns[row][0]->_index_maps[0]);
    std::int32_t row_size = patterns[row][0]->_index_maps[0]->size_local();
    const int bs0 = patterns[row][0]->_index_maps[0]->block_size;

    // FIXME: Issue somewhere here when block size > 1
    assert(bs0 * row_size
           == (std::int32_t)patterns[row][0]->_diagonal->num_nodes());
    diagonal.resize(diagonal.size() + bs0 * row_size);
    assert(bs0 * row_size
           == (std::int32_t)patterns[row][0]->_off_diagonal->num_nodes());
    off_diagonal.resize(off_diagonal.size() + bs0 * row_size);

    // Iterate over block columns of current block row
    std::int64_t col_global_offset = col_process_offset;
    for (std::size_t col = 0; col < patterns[row].size(); ++col)
    {
      // Get pattern for this block
      auto p = patterns[row][col];
      assert(p);

      // Check that
      if (!p->_diagonal)
      {
        throw std::runtime_error("Sub-sparsity pattern has not been finalised "
                                 "(assemble needs to be called)");
      }

      auto index_map1 = p->index_map(1);
      assert(index_map1);
      for (int k = 0; k < p->_diagonal->num_nodes(); ++k)
      {
        // Diagonal block
        auto edges0 = p->_diagonal->links(k);

        // for (std::size_t c : edges0)
        for (Eigen::Index i = 0; i < edges0.rows(); ++i)
        {
          // Get local index and convert to global (for this block)
          std::int32_t c = edges0[i];
          const std::int64_t J = col_map(c, *index_map1);
          assert(J >= 0);
          // const int rank = MPI::rank(MPI_COMM_WORLD);
          // assert(index_map1->owner(J / index_map1->block_size) == rank);

          // Get new index
          const std::int64_t offset = fem::get_global_offset(cmaps, col, J);
          const std::int64_t c_new = J + offset - col_process_offset;
          assert(c_new >= 0);
          diagonal[k + row_local_offset].push_back(c_new);
        }

        // Off-diagonal block
        auto edges1 = p->_off_diagonal->links(k);
        for (Eigen::Index i = 0; i < edges1.rows(); ++i)
        {
          const std::int64_t c = edges1[i];
          // Get new index
          const std::int64_t offset = fem::get_global_offset(cmaps, col, c);
          off_diagonal[k + row_local_offset].push_back(c + offset);
        }
      }

      // Increment global column offset
      col_global_offset
          += p->_index_maps[1]->size_local() * p->_index_maps[1]->block_size;
    }

    // Increment local row offset
    row_local_offset += bs0 * row_size;
  }

  // FIXME: Need to add unowned entries?

  // Initialise common::IndexMaps for merged pattern
  auto p00 = patterns[0][0];
  assert(p00);
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> ghosts;
  _index_maps[0] = std::make_shared<common::IndexMap>(
      p00->mpi_comm(), row_local_size, ghosts, 1);
  _index_maps[1] = std::make_shared<common::IndexMap>(
      p00->mpi_comm(), col_local_size, ghosts, 1);

  // TODO: Is the erase step required here, or will there be no
  // duplicates?
  for (auto& row : diagonal)
  {
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
  }
  _diagonal = std::make_shared<graph::AdjacencyList<std::int32_t>>(diagonal);

  for (auto& row : off_diagonal)
  {
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
  }
  _off_diagonal
      = std::make_shared<graph::AdjacencyList<std::int64_t>>(off_diagonal);
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> SparsityPattern::local_range(int dim) const
{
  const int bs = _index_maps.at(dim)->block_size;
  const std::array<std::int64_t, 2> lrange = _index_maps[dim]->local_range();
  return {{bs * lrange[0], bs * lrange[1]}};
}
//-----------------------------------------------------------------------------
std::shared_ptr<const common::IndexMap>
SparsityPattern::index_map(int dim) const
{
  return _index_maps.at(dim);
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert(
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& rows,
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& cols)
{
  if (_diagonal)
  {
    throw std::runtime_error(
        "Cannot insert into sparsity pattern. It has already been assembled");
  }

  const common::IndexMap& index_map0 = *_index_maps[0];
  const int bs0 = index_map0.block_size;
  const std::int32_t local_size0
      = bs0 * (index_map0.size_local() + index_map0.num_ghosts());

  const common::IndexMap& index_map1 = *_index_maps[1];
  const int bs1 = index_map1.block_size;
  const std::int32_t local_size1 = bs1 * index_map1.size_local();

  for (Eigen::Index i = 0; i < rows.rows(); ++i)
  {
    if (rows[i] < local_size0)
    {
      for (Eigen::Index j = 0; j < cols.rows(); ++j)
      {
        if (cols[j] < local_size1)
          _diagonal_cache[rows[i]].push_back(cols[j]);
        else
        {
          const std::int64_t J = col_map(cols[j], index_map1);
          _off_diagonal_cache[rows[i]].push_back(J);
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
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& rows)
{
  if (_diagonal)
  {
    throw std::runtime_error(
        "Cannot insert into sparsity pattern. It has already been assembled");
  }

  const common::IndexMap& index_map0 = *_index_maps[0];
  const int bs0 = index_map0.block_size;
  const std::int32_t local_size0
      = bs0 * (index_map0.size_local() + index_map0.num_ghosts());

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
  const int bs0 = _index_maps[0]->block_size;
  const std::int32_t local_size0 = _index_maps[0]->size_local();
  const std::int32_t num_ghosts0 = _index_maps[0]->num_ghosts();
  const std::array<std::int64_t, 2> local_range0
      = _index_maps[0]->local_range();

  assert(_index_maps[1]);
  const int bs1 = _index_maps[1]->block_size;
  const std::array<std::int64_t, 2> local_range1
      = _index_maps[1]->local_range();

  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& ghosts
      = _index_maps[0]->ghosts();

  // For each ghost row, pack and send (global row, global col) pairs to
  // send to neighborhood
  std::vector<std::int64_t> ghost_data;
  for (int i = 0; i < num_ghosts0; ++i)
  {
    const std::int64_t row_node = ghosts[i];
    const std::int32_t row_node_local = local_size0 + i;
    for (int j = 0; j < bs0; ++j)
    {
      const std::int64_t row = bs0 * row_node + j;
      const std::int32_t row_local = bs0 * row_node_local + j;
      assert((std::size_t)row_local < _diagonal_cache.size());
      auto cols = _diagonal_cache[row_local];
      for (std::size_t c = 0; c < cols.size(); ++c)
      {
        ghost_data.push_back(row);

        // Convert to global column index
        const std::int64_t J = col_map(cols[c], *_index_maps[1]);
        ghost_data.push_back(J);
      }
      auto cols_off = _off_diagonal_cache[row_local];
      for (std::size_t c = 0; c < cols_off.size(); ++c)
      {
        ghost_data.push_back(row);
        ghost_data.push_back(cols_off[c]);
      }
    }
  }

  // Get number of processes in neighbourhood
  MPI_Comm comm = _index_maps[0]->mpi_comm_neighborhood();
  int num_neighbours(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(comm, &num_neighbours, &outdegree, &weighted);
  assert(num_neighbours == outdegree);

  // Figure out how much data to receive from each neighbour
  const int num_my_rows = ghost_data.size();
  std::vector<int> num_rows_recv(num_neighbours);
  MPI_Neighbor_allgather(&num_my_rows, 1, MPI_INT, num_rows_recv.data(), 1,
                         MPI_INT, comm);

  // Compute displacements for data to receive
  std::vector<int> disp(num_neighbours + 1, 0);
  std::partial_sum(num_rows_recv.begin(), num_rows_recv.end(),
                   disp.begin() + 1);

  // NOTE: Send unowned rows to all neighbours could be a bit 'lazy' and
  // MPI_Neighbor_alltoallv could be used to send just to the owner, but
  // maybe the number of rows exchanged in the neighbourhood are
  // relatively small that MPI_Neighbor_allgatherv is simpler.

  // Send all unowned rows to neighbours, and receive rows from
  // neighbours
  std::vector<std::int64_t> ghost_data_received(disp.back());
  MPI_Neighbor_allgatherv(ghost_data.data(), ghost_data.size(), MPI_INT64_T,
                          ghost_data_received.data(), num_rows_recv.data(),
                          disp.data(), MPI_INT64_T, comm);

  // Add data received from the neighbourhood
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
  for (auto& row : _diagonal_cache)
  {
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
  }
  _diagonal
      = std::make_shared<graph::AdjacencyList<std::int32_t>>(_diagonal_cache);
  std::vector<std::vector<std::int32_t>>().swap(_diagonal_cache);

  _off_diagonal_cache.resize(bs0 * local_size0);
  for (auto& row : _off_diagonal_cache)
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
Eigen::Array<std::int32_t, Eigen::Dynamic, 1>
SparsityPattern::num_nonzeros_diagonal() const
{
  if (!_diagonal)
    throw std::runtime_error("Sparsity pattern has not been finalised.");

  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> num_nonzeros(
      _diagonal->num_nodes());
  for (int i = 0; i < _diagonal->num_nodes(); ++i)
    num_nonzeros[i] = _diagonal->num_links(i);

  return num_nonzeros;
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int32_t, Eigen::Dynamic, 1>
SparsityPattern::num_nonzeros_off_diagonal() const
{
  if (!_off_diagonal)
    throw std::runtime_error("Sparsity pattern has not been finalised.");

  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> num_nonzeros(
      _off_diagonal->num_nodes());
  for (int i = 0; i < _off_diagonal->num_nodes(); ++i)
    num_nonzeros[i] = _off_diagonal->num_links(i);

  return num_nonzeros;
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int32_t, Eigen::Dynamic, 1>
SparsityPattern::num_local_nonzeros() const
{
  return num_nonzeros_diagonal() + num_nonzeros_off_diagonal();
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
std::string SparsityPattern::str() const
{
  if (!_diagonal)
    throw std::runtime_error("Sparsity pattern has not been assembled.");
  assert(_off_diagonal);

  // Print each row
  std::stringstream s;
  assert(_off_diagonal->num_nodes() == _diagonal->num_nodes());
  for (int i = 0; i < _diagonal->num_nodes(); i++)
  {
    s << "Row " << i << ":";
    s << " " << _diagonal->links(i);
    s << " " << _off_diagonal->links(i);
    s << std::endl;
  }

  return s.str();
}
//-----------------------------------------------------------------------------
