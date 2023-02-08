// Copyright (C) 2007-2020 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/MPI.h>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace dolfinx::graph
{
template <typename T>
class AdjacencyList;
}

namespace dolfinx::common
{
class IndexMap;
}

namespace dolfinx::la
{

/// This class provides a sparsity pattern data structure that can be
/// used to initialize sparse matrices. After assembly, column indices
/// are always sorted in increasing order. Ghost entries are kept after
/// assembly.
class SparsityPattern
{

public:
  /// Create an empty sparsity pattern with specified dimensions
  /// @param[in] comm The communicator that the pattenr is defined on
  /// @param[in] maps The index maps describing the [0] row and [1]
  /// column index ranges (up to a block size)
  /// @param[in] bs The block sizes for the [0] row and [1] column maps
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
  ///   patterns must not be finalised. Null block are permitted
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

  /// Insert non-zero locations using local (process-wise) indices
  void insert(const std::span<const std::int32_t>& rows,
              const std::span<const std::int32_t>& cols);

  /// Insert non-zero locations on the diagonal
  /// @param[in] rows The rows in local (process-wise) indices. The
  /// indices must exist in the row IndexMap.
  void insert_diagonal(std::span<const std::int32_t> rows);

  /// Finalize sparsity pattern and communicate off-process entries
  void assemble();

  /// Index map for given dimension dimension. Returns the index map for
  /// rows and columns that will be set by the current MPI rank.
  /// @param[in] dim The requested map, row (0) or column (1)
  /// @return The index map
  std::shared_ptr<const common::IndexMap> index_map(int dim) const;

  /// Global indices of non-zero columns on owned rows
  ///
  /// @note The ghosts are computed only once SparsityPattern::assemble
  /// has been called
  /// @return The global index non-zero columns on this process,
  /// including ghosts
  std::vector<std::int64_t> column_indices() const;

  /// Builds the index map for columns after assembly of the sparsity
  /// pattern
  /// @return Map for all non-zero columns on this process, including
  /// ghosts
  /// @todo Should this be compted and stored when finalising the
  /// SparsityPattern?
  common::IndexMap column_index_map() const;

  /// Return index map block size for dimension dim
  int block_size(int dim) const;

  /// Number of nonzeros on this rank after assembly, including ghost
  /// rows.
  std::int64_t num_nonzeros() const;

  /// Number of non-zeros in owned columns (diagonal block) on a given row
  /// @note Can also be used on ghost rows
  std::int32_t nnz_diag(std::int32_t row) const;

  /// Number of non-zeros in unowned columns (off-diagonal block) on a
  /// given row
  /// @note Can also be used on ghost rows
  std::int32_t nnz_off_diag(std::int32_t row) const;

  /// Sparsity pattern graph after assembly. Uses local indices for the
  /// columns.
  /// @note Column global indices can be obtained from
  /// SparsityPattern::column_index_map()
  /// @note Includes ghost rows
  const graph::AdjacencyList<std::int32_t>& graph() const;

  /// Row-wise start of off-diagonal (unowned columns) on each row
  /// @note Includes ghost rows
  std::span<const int> off_diagonal_offset() const;

  /// Return MPI communicator
  MPI_Comm comm() const;

private:
  // MPI communicator
  dolfinx::MPI::Comm _comm;

  // Index maps for each dimension
  std::array<std::shared_ptr<const common::IndexMap>, 2> _index_maps;

  // Block size
  std::array<int, 2> _bs;

  // Non-zero ghost columns in owned rows
  std::vector<std::int64_t> _col_ghosts;
  // Owning process of ghost columns in owned rows
  std::vector<std::int32_t> _col_ghost_owners;

  // Cache for unassembled entries on owned and unowned (ghost) rows
  std::vector<std::vector<std::int32_t>> _row_cache;

  // Sparsity pattern data (computed once pattern is finalised)
  std::shared_ptr<graph::AdjacencyList<std::int32_t>> _graph;

  // Start of off-diagonal (unowned columns) on each row
  std::vector<int> _off_diagonal_offset;
};
} // namespace dolfinx::la
