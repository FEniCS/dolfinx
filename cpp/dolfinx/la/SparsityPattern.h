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

namespace dolfinx::common
{
class IndexMap;
}

namespace dolfinx::la
{
/// Sparsity pattern data structure that can be used to initialize
/// sparse matrices. After assembly, column indices are always sorted in
/// increasing order. Ghost entries are kept after assembly.
class SparsityPattern
{
public:
  /// @brief Create an empty sparsity pattern with specified dimensions.
  /// @param[in] comm Communicator that the pattern is defined on.
  /// @param[in] maps Index maps describing the [0] row and [1] column
  /// index ranges (up to a block size).
  /// @param[in] bs Block sizes for the [0] row and [1] column maps.
  SparsityPattern(
      MPI_Comm comm,
      const std::array<std::shared_ptr<const common::IndexMap>, 2>& maps,
      const std::array<int, 2>& bs);

  /// Create a new sparsity pattern by concatenating sub-patterns, e.g.
  /// pattern =[ pattern00 ][ pattern 01]
  ///          [ pattern10 ][ pattern 11]
  ///
  /// @param[in] comm Communicator that the pattern is defined on.
  /// @param[in] patterns Rectangular array of sparsity pattern. The
  /// patterns must not be finalised. Null block are permitted/
  /// @param[in] maps Pairs of (index map, block size) for each row
  /// block (maps[0]) and column blocks (maps[1])/
  /// @param[in] bs Block sizes for the sparsity pattern entries/
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

  /// @brief Insert non-zero locations using local (process-wise)
  /// indices.
  /// @param[in] row local row index
  /// @param[in] col local column index
  void insert(std::int32_t row, std::int32_t col);

  /// @brief Insert non-zero locations using local (process-wise)
  /// indices.
  ///
  /// This routine inserts non-zero locations at the outer product of rows and
  /// cols into the sparsity pattern, i.e. adds the matrix entries at
  ///   A[row[i], col[j]] for all i, j.
  ///
  /// @param[in] rows list of the local row indices
  /// @param[in] cols list of the local column indices
  void insert(std::span<const std::int32_t> rows,
              std::span<const std::int32_t> cols);

  /// @brief Insert non-zero locations on the diagonal
  /// @param[in] rows Rows in local (process-wise) indices. The indices
  /// must exist in the row IndexMap.
  void insert_diagonal(std::span<const std::int32_t> rows);

  /// @brief Finalize sparsity pattern and communicate off-process entries
  void finalize();

  /// @brief Index map for given dimension dimension. Returns the index
  /// map for rows and columns that will be set by the current MPI rank.
  /// @param[in] dim Requested map, row (0) or column (1).
  /// @return The index map.
  std::shared_ptr<const common::IndexMap> index_map(int dim) const;

  /// @brief Global indices of non-zero columns on owned rows.
  ///
  /// @note The ghosts are computed only once SparsityPattern::finalize
  /// has been called.
  /// @return Global index non-zero columns on this process, including
  /// ghosts.
  std::vector<std::int64_t> column_indices() const;

  /// @brief Builds the index map for columns after assembly of the
  /// sparsity pattern
  /// @return Map for all non-zero columns on this process, including
  /// ghosts
  /// @todo Should this be compted and stored when finalising the
  /// SparsityPattern?
  common::IndexMap column_index_map() const;

  /// @brief Return index map block size for dimension dim
  int block_size(int dim) const;

  /// @brief Number of nonzeros on this rank after assembly, including
  /// ghost rows.
  std::int64_t num_nonzeros() const;

  /// @brief Number of non-zeros in owned columns (diagonal block) on a
  /// given row.
  /// @note Can also be used on ghost rows
  std::int32_t nnz_diag(std::int32_t row) const;

  /// @brief Number of non-zeros in unowned columns (off-diagonal block)
  /// on a given row.
  /// @note Can also be used on ghost rows
  std::int32_t nnz_off_diag(std::int32_t row) const;

  /// @brief Sparsity pattern graph after assembly. Uses local indices
  /// for the columns.
  /// @note Column global indices can be obtained from
  /// SparsityPattern::column_index_map()
  /// @note Includes ghost rows
  /// @return Adjacency list edges and offsets
  std::pair<std::span<const std::int32_t>, std::span<const std::int64_t>>
  graph() const;

  /// @brief Row-wise start of off-diagonals (unowned columns) for each
  /// row.
  /// @note Includes ghost rows
  std::span<const std::int32_t> off_diagonal_offsets() const;

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

  // Sparsity pattern adjacency data (computed once pattern is
  // finalised). _edges holds the edges (connected dofs). The edges for
  // node i are in the range [_offsets[i], _offsets[i + 1]).
  std::vector<std::int32_t> _edges;
  std::vector<std::int64_t> _offsets;

  // Start of off-diagonal (unowned columns) on each row (row-wise)
  std::vector<std::int32_t> _off_diagonal_offsets;
};
} // namespace dolfinx::la
