// Copyright (C) 2007-2020 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/MPI.h>
#include <dolfinx/common/span.hpp>
#include <memory>
#include <utility>
#include <vector>

namespace dolfinx
{

namespace graph
{
template <typename T>
class AdjacencyList;
}

namespace common
{
class IndexMap;
}

namespace la
{

/// This class provides a sparsity pattern data structure that can be
/// used to initialize sparse matrices.

class SparsityPattern
{

public:
  /// Create an empty sparsity pattern with specified dimensions
  SparsityPattern(
      MPI_Comm comm,
      const std::array<std::shared_ptr<const common::IndexMap>, 2>& maps,
      const std::array<int, 2>& bs);

  /// Create a new sparsity pattern by concatenating sub-patterns, e.g.
  /// pattern =[ pattern00 ][ pattern 01]
  ///          [ pattern10 ][ pattern 11]
  ///
  /// @param[in] comm The MPI communicator
  /// @param[in] patterns Rectangular array of sparsity pattern. The
  ///   patterns must not be finalised. Null block are permited
  /// @param[in] maps Pairs of (index map, block size) for each row
  ///   block (maps[0]) and column blocks (maps[1])
  /// @param[in] bs Block sizes for the sparsity pattern entries
  SparsityPattern(
      MPI_Comm comm,
      const std::vector<std::vector<const SparsityPattern*>>& patterns,
      const std::array<
          std::vector<
              std::pair<std::reference_wrapper<const common::IndexMap>, int>>,
          2>& maps,
      const std::array<std::vector<int>, 2>& bs);

  SparsityPattern(const SparsityPattern& pattern) = delete;

  /// Move constructor
  SparsityPattern(SparsityPattern&& pattern) = default;

  /// Destructor
  ~SparsityPattern() = default;

  /// Move assignment
  SparsityPattern& operator=(SparsityPattern&& pattern) = default;

  /// Return index map for dimension dim
  std::shared_ptr<const common::IndexMap> index_map(int dim) const;

  /// Return index map block size for dimension dim
  int block_size(int dim) const;

  /// Insert non-zero locations using local (process-wise) indices
  void insert(const tcb::span<const std::int32_t>& rows,
              const tcb::span<const std::int32_t>& cols);

  /// Insert non-zero locations on the diagonal
  /// @param[in] rows The rows in local (process-wise) indices. The
  ///   indices must exist in the row IndexMap.
  void insert_diagonal(const std::vector<std::int32_t>& rows);

  /// Finalize sparsity pattern and communicate off-process entries
  void assemble();

  /// Return number of local nonzeros
  std::int64_t num_nonzeros() const;

  /// Sparsity pattern for the owned (diagonal) block. Uses local
  /// indices for the columns.
  const graph::AdjacencyList<std::int32_t>& diagonal_pattern() const;

  /// Sparsity pattern for the un-owned (off-diagonal) columns. Uses local
  /// indices for the columns. Translate to global with column IndexMap.
  const graph::AdjacencyList<std::int32_t>& off_diagonal_pattern() const;

  /// Return MPI communicator
  MPI_Comm mpi_comm() const;

private:
  // MPI communicator
  dolfinx::MPI::Comm _mpi_comm;

  // common::IndexMaps for each dimension
  std::array<std::shared_ptr<const common::IndexMap>, 2> _index_maps;
  std::array<int, 2> _bs;

  // Caches for unassembled entries on owned and unowned (ghost) rows
  std::vector<std::vector<std::int32_t>> _cache_owned;
  std::vector<std::vector<std::int32_t>> _cache_unowned;

  // Sparsity pattern data (computed once pattern is finalised)
  std::shared_ptr<graph::AdjacencyList<std::int32_t>> _diagonal;
  std::shared_ptr<graph::AdjacencyList<std::int32_t>> _off_diagonal;
};
} // namespace la
} // namespace dolfinx
