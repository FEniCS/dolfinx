// Copyright (C) 2007-2018 Garth N. Wells
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
      = index_maps[0]->block_size * index_maps[0]->size_local();
  _diagonal.resize(local_size0);
  _off_diagonal.resize(local_size0);
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

  const bool distributed = MPI::size(comm) > 1;

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
    assert(bs0 * row_size == (std::int32_t)patterns[row][0]->_diagonal.size());
    // if (!patterns[row][0]->_diagonal.empty())
    // {
    //   if (row_size != patterns[row][0]->_diagonal.size())
    //   {
    //     throw std::runtime_error("Mismtach between SparsityPattern size "
    //                              "(diagonal) and row range map.");
    //   }
    // }

    this->_diagonal.resize(this->_diagonal.size() + bs0 * row_size);
    if (distributed)
    {
      assert(bs0 * row_size
             == (std::int32_t)patterns[row][0]->_off_diagonal.size());
      this->_off_diagonal.resize(this->_off_diagonal.size() + bs0 * row_size);
    }

    // Iterate over block columns of current block row
    std::int64_t col_global_offset = col_process_offset;
    for (std::size_t col = 0; col < patterns[row].size(); ++col)
    {
      // Get pattern for this block
      auto p = patterns[row][col];
      assert(p);

      // Check that
      if (!p->_non_local.empty())
      {
        throw std::runtime_error("Sub-sparsity pattern has not been finalised "
                                 "(apply needs to be called)");
      }

      for (std::size_t k = 0; k < p->_diagonal.size(); ++k)
      {
        // Diagonal block
        std::vector<std::size_t> edges0 = p->_diagonal[k].set();
        // std::transform(edges0.begin(), edges0.end(), edges0.begin(),
        //                std::bind2nd(std::plus<double>(), col_global_offset));
        // assert(k + row_local_offset < this->_diagonal.size());
        // this->_diagonal[k + row_local_offset].insert(edges0.begin(),
        //                                              edges0.end());

        for (std::size_t c : edges0)
        {
          // Get new index
          std::int64_t c_new = fem::get_global_index(cmaps, col, c);
          this->_diagonal[k + row_local_offset].insert(c_new);
        }

        // Off-diagonal block
        if (distributed)
        {
          std::vector<std::size_t> edges1 = p->_off_diagonal[k].set();
          for (std::size_t c : edges1)
          {
            // Get new index
            std::int64_t c_new = fem::get_global_index(cmaps, col, c);
            this->_off_diagonal[k + row_local_offset].insert(c_new);
          }
          // std::transform(edges1.begin(), edges1.end(), edges1.begin(),
          //                std::bind2nd(std::plus<double>(),
          //                col_global_offset));
          // assert(k + row_local_offset < this->_off_diagonal.size());
          // this->_off_diagonal[k + row_local_offset].insert(edges1.begin(),
          //                                                  edges1.end());
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
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert(
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& rows,
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& cols)
{
  const common::IndexMap& index_map0 = *_index_maps[0];
  const common::IndexMap& index_map1 = *_index_maps[1];

  const int bs0 = index_map0.block_size;
  const std::int32_t local_size0 = bs0 * index_map0.size_local();

  const int bs1 = index_map1.block_size;
  const auto local_range1 = index_map1.local_range();

  // Parallel mode, use either diagonal, off_diagonal, non_local or
  // full_rows
  for (Eigen::Index i = 0; i < rows.rows(); ++i)
  {
    if (rows[i] < local_size0)
    {
      // Store local entry in diagonal or off-diagonal block
      for (Eigen::Index j = 0; j < cols.rows(); ++j)
      {
        const auto J = col_map(cols[j], index_map1);
        if ((bs1 * local_range1[0]) <= J and J < (bs1 * local_range1[1]))
        {
          assert(rows[i] < (PetscInt)_diagonal.size());
          _diagonal[rows[i]].insert(J);
        }
        else
        {
          assert(rows[i] < (PetscInt)_off_diagonal.size());
          _off_diagonal[rows[i]].insert(J);
        }
      }
    }
    else
    {
      // Store non-local entry (communicated later during assemble())
      for (Eigen::Index j = 0; j < cols.rows(); ++j)
      {
        _non_local.push_back(rows[i]);
        const auto J = col_map(cols[j], index_map1);
        _non_local.push_back(J);
      }
    }
  }
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> SparsityPattern::local_range(int dim) const
{
  assert(dim < 2);
  const int bs = _index_maps[dim]->block_size;
  auto lrange = _index_maps[dim]->local_range();
  return {{bs * lrange[0], bs * lrange[1]}};
}
//-----------------------------------------------------------------------------
std::shared_ptr<const common::IndexMap>
SparsityPattern::index_map(int dim) const
{
  assert(dim < 2);
  return _index_maps[dim];
}
//-----------------------------------------------------------------------------
std::size_t SparsityPattern::num_nonzeros() const
{
  std::size_t nz = 0;
  for (const auto& slice : _diagonal)
    nz += slice.size();
  for (const auto& slice : _off_diagonal)
    nz += slice.size();

  return nz;
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int32_t, Eigen::Dynamic, 1>
SparsityPattern::num_nonzeros_diagonal() const
{
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> num_nonzeros(_diagonal.size());
  for (auto slice = _diagonal.begin(); slice != _diagonal.end(); ++slice)
    num_nonzeros[slice - _diagonal.begin()] = slice->size();

  return num_nonzeros;
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int32_t, Eigen::Dynamic, 1>
SparsityPattern::num_nonzeros_off_diagonal() const
{
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> num_nonzeros(
      _off_diagonal.size());

  // Return if there is no off-diagonal
  if (_off_diagonal.empty())
  {
    num_nonzeros.setZero();
    return num_nonzeros;
  }

  // Compute number of nonzeros per generalised row
  for (auto slice = _off_diagonal.begin(); slice != _off_diagonal.end();
       ++slice)
  {
    num_nonzeros[slice - _off_diagonal.begin()] = slice->size();
  }

  return num_nonzeros;
}
//-----------------------------------------------------------------------------
Eigen::Array<std::int32_t, Eigen::Dynamic, 1>
SparsityPattern::num_local_nonzeros() const
{
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> num_nonzeros
      = num_nonzeros_diagonal();
  if (!_off_diagonal.empty())
  {
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> num_nonzeros_off_diag
        = num_nonzeros_off_diagonal();
    num_nonzeros += num_nonzeros_off_diag;
  }

  return num_nonzeros;
}
//-----------------------------------------------------------------------------
void SparsityPattern::assemble()
{
  const int bs0 = _index_maps[0]->block_size;
  const int bs1 = _index_maps[1]->block_size;
  const std::array<std::int64_t, 2> local_range0
      = _index_maps[0]->local_range();
  const std::array<std::int64_t, 2> local_range1
      = _index_maps[1]->local_range();
  const std::int32_t local_size0 = bs0 * _index_maps[0]->size_local();
  const std::int64_t offset0 = bs0 * local_range0[0];

  // Figure out correct process for each non-local entry
  assert(_non_local.size() % 2 == 0);

  // Get local-to-global for unowned (ghost) blocks
  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& local_to_global
      = _index_maps[0]->ghosts();

  // Pack off-process row data to send
  std::vector<std::int64_t> non_local_send;
  for (std::size_t i = 0; i < _non_local.size(); i += 2)
  {
    // Get local indices of off-process dofs
    const std::int64_t i_index = _non_local[i];

    if (i_index < local_size0)
      throw std::runtime_error("Should not reach this point.");
    else
    {
      // Compute global I index from i_index
      const int i_local = i_index - local_size0;
      const std::div_t div = std::div(i_local, bs0);
      const int i_node = div.quot;
      const int i_component = div.rem;

      const std::int64_t I_node = local_to_global[i_node];
      const std::int64_t I = bs0 * I_node + i_component;

      // Add to send buffer
      const std::int64_t J = _non_local[i + 1];
      non_local_send.push_back(I);
      non_local_send.push_back(J);
    }
  }

  // Get number of processes in neighbourhood
  MPI_Comm comm = _index_maps[0]->mpi_comm_neighborhood();
  int num_neighbours(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(comm, &num_neighbours, &outdegree, &weighted);
  assert(num_neighbours == outdegree);

  // Figure out how many rows to receive from each neighbour
  const int num_my_rows = non_local_send.size();
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
  std::vector<std::int64_t> non_local_received(disp.back());
  MPI_Neighbor_allgatherv(non_local_send.data(), non_local_send.size(),
                          MPI_INT64_T, non_local_received.data(),
                          num_rows_recv.data(), disp.data(), MPI_INT64_T, comm);

  // Insert non-local entries received from other processes
  assert(non_local_received.size() % 2 == 0);
  for (std::size_t i = 0; i < non_local_received.size(); i += 2)
  {
    // Get global row and column
    const std::int64_t I = non_local_received[i];
    const std::int64_t J = non_local_received[i + 1];

    // Check if I own row I
    if (I >= bs0 * local_range0[0] and I < bs0 * local_range0[1])
    {
      const std::int32_t i_index = I - offset0;
      if (J >= bs1 * local_range1[0] and J < bs1 * local_range1[1])
      {
        assert(i_index < (std::int32_t)_diagonal.size());
        _diagonal[i_index].insert(J);
      }
      else
      {
        assert(i_index < (std::int32_t)_off_diagonal.size());
        _off_diagonal[i_index].insert(J);
      }
    }
  }

  // Clear non-local entries
  _non_local.clear();
}
//-----------------------------------------------------------------------------
std::string SparsityPattern::str() const
{
  // Print each row
  std::stringstream s;
  for (std::size_t i = 0; i < _diagonal.size(); i++)
  {
    s << "Row " << i << ":";

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

  std::size_t bs0 = _index_maps[0]->block_size;
  std::size_t size0 = bs0 * _index_maps[0]->size_global();

  std::size_t bs1 = _index_maps[1]->block_size;
  std::size_t size1 = bs1 * _index_maps[1]->size_global();

  // Return number of entries
  std::cout << "Matrix of size " << size0 << " x " << size1 << " has "
            << num_nonzeros_total << " ("
            << 100.0 * num_nonzeros_total / (size0 * size1) << "%)"
            << " nonzero entries." << std::endl;
  if (num_nonzeros_total != num_nonzeros_diagonal)
  {
    std::cout << "Diagonal: " << num_nonzeros_diagonal << " ("
              << (100.0 * static_cast<double>(num_nonzeros_diagonal)
                  / static_cast<double>(num_nonzeros_total))
              << "%), ";
    std::cout << "off-diagonal: " << num_nonzeros_off_diagonal << " ("
              << (100.0 * static_cast<double>(num_nonzeros_off_diagonal)
                  / static_cast<double>(num_nonzeros_total))
              << "%), ";
    std::cout << "non-local: " << num_nonzeros_non_local << " ("
              << (100.0 * static_cast<double>(num_nonzeros_non_local)
                  / static_cast<double>(num_nonzeros_total))
              << "%)";
    std::cout << std::endl;
  }
}
//-----------------------------------------------------------------------------
