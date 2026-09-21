// Copyright (C) 2007-2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstddef>
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
  SparsityPattern(MPI_Comm comm,
                  std::array<std::shared_ptr<const common::IndexMap>, 2> maps,
                  std::array<int, 2> bs);

  /// @brief Create a new sparsity pattern by concatenating sub-patterns,
  /// e.g.
  /// pattern =[ pattern00 ][ pattern 01]
  ///          [ pattern10 ][ pattern 11]
  ///
  /// @param[in] comm Communicator that the pattern is defined on.
  /// @param[in] patterns Rectangular array of sparsity pattern. The
  /// patterns must not be finalised. Null blocks are permitted.
  /// @param[in] maps Pairs of (index map, block size) for each row
  /// block (maps[0]) and column blocks (maps[1]).
  /// @param[in] bs Block sizes for the sparsity pattern entries.
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

  /// @brief Reserve storage for additional insertions.
  /// @param[in] num_entries Expected number of (row, column) entries,
  /// including duplicates, to insert in addition to those already cached.
  void reserve(std::size_t num_entries);

  /// @brief Reserve storage for additional `insert(rows, cols)` calls.
  ///
  /// Blocks are cached in the form they are inserted in, so the number
  /// of calls and the total number of row and column indices are needed
  /// rather than the number of (row, column) entries.
  ///
  /// @param[in] num_blocks Number of `insert(rows, cols)` calls.
  /// @param[in] num_rows Total number of row indices over those calls.
  /// @param[in] num_cols Total number of column indices over those
  /// calls.
  void reserve_blocks(std::size_t num_blocks, std::size_t num_rows,
                      std::size_t num_cols);

  /// @brief Insert non-zero locations using local (process-wise)
  /// indices.
  /// @param[in] row local row index
  /// @param[in] col local column index
  void insert(std::int32_t row, std::int32_t col);

  /// @brief Insert non-zero locations using local (process-wise)
  /// indices.
  ///
  /// This routine inserts non-zero locations at the outer product of
  /// rows and cols into the sparsity pattern, i.e. adds the matrix
  /// entries at `A[row[i], col[j]] for all i, j`.
  ///
  /// @param[in] rows list of the local row indices
  /// @param[in] cols list of the local column indices
  void insert(std::span<const std::int32_t> rows,
              std::span<const std::int32_t> cols);

  /// @brief Insert non-zero locations on the diagonal
  /// @param[in] rows Rows in local (process-wise) indices. The indices
  /// must exist in the row IndexMap.
  void insert_diagonal(std::span<const std::int32_t> rows);

  /// @brief Finalize sparsity pattern and communicate off-process
  /// entries
  void finalize();

  /// @brief Index map for given dimension. Returns the index
  /// map for rows and columns that will be set by the current MPI rank.
  /// @note After finalization, the column index map is updated to account for
  /// additional column entries from other processes.
  /// @param[in] dim Requested map, row (0) or column (1).
  /// @return The index map.
  std::shared_ptr<const common::IndexMap> index_map(int dim) const;

  /// @brief Global column indices corresponding to the local column
  /// indices used by SparsityPattern::graph.
  ///
  /// Entry `i` of the returned vector is the global index of local
  /// column `i`: for `i` in the owned range this is every owned
  /// column (whether or not it holds a non-zero entry), and for `i`
  /// beyond the owned range it is the ghost column, i.e. a column
  /// with at least one non-zero entry owned by another rank.
  ///
  /// @note The ghosts are computed only once SparsityPattern::finalize
  /// has been called.
  /// @return Global column indices on this process, including ghosts.
  std::vector<std::int64_t> column_indices() const;

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
  /// SparsityPattern::column_indices()
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

  // Cache of unassembled entries on owned and unowned (ghost) rows,
  // held until finalize().
  //
  // insert(rows, cols) inserts the outer product of `rows` and `cols`,
  // so storing one (row, column) pair per entry repeats each index
  // rows.size() or cols.size() times over. Blocks are instead kept in
  // the form they arrive in and expanded only in finalize(): for a P1
  // tetrahedron that is 4 + 4 indices per cell rather than 16 + 16.
  std::vector<std::int32_t> _cache_brows, _cache_bcols;
  std::vector<std::int64_t> _cache_boffs_r{0}, _cache_boffs_c{0};

  // Cache of individually inserted (row, column) pairs (row-major COO)
  std::vector<std::int32_t> _cache_rows;
  std::vector<std::int32_t> _cache_cols;

  // Rows with a cached diagonal entry, from insert_diagonal
  std::vector<std::int32_t> _cache_diag;

  /// @brief Total number of cached (row, column) entries, counting
  /// duplicates.
  std::size_t num_cached() const;

  /// @brief Expand every cached entry and group the columns by row.
  /// @param[in] num_rows Number of rows to bucket into.
  /// @param[in] num_cols Number of columns in the cached index space.
  /// @return Row offsets (size `num_rows + 1`) and columns grouped by
  /// row and deduplicated, but not sorted, within each row.
  std::pair<std::vector<std::int64_t>, std::vector<std::int32_t>>
  bucket_cache(std::int32_t num_rows, std::int32_t num_cols) const;

  // Sparsity pattern adjacency data (computed once pattern is
  // finalised). _edges holds the edges (connected dofs). The edges for
  // node i are in the range [_offsets[i], _offsets[i + 1]).
  std::vector<std::int32_t> _edges;
  std::vector<std::int64_t> _offsets;

  // Start of off-diagonal (unowned columns) on each row (row-wise)
  std::vector<std::int32_t> _off_diagonal_offsets;
};
} // namespace dolfinx::la
