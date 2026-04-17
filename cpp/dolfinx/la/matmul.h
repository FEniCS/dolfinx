// Copyright (C) 2026 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/IndexMap.h>

#include <dolfinx/la/MatrixCSR.h>
#include <mpi.h>

#include <algorithm>
#include <numeric>
#include <span>
#include <vector>

namespace dolfinx::la
{

/// @brief Fetch the rows of B that correspond to the ghost columns of A.
///
/// For computing the product A*B, each rank needs the rows of B whose
/// global indices match the ghost columns of A. Those ghost columns are
/// owned by remote ranks, so we request those rows via neighbourhood
/// communication.
///
/// @param A Matrix whose ghost column indices determine which rows of B
///          are needed.
/// @param B Matrix whose rows are fetched.
/// @return A new MatrixCSR containing the fetched ghost rows of B, with
///         an extended column IndexMap covering any new ghost columns
///         introduced by those rows.
namespace impl
{

/// @brief Lightweight sparsity descriptor satisfying the
///        `SparsityImplementation` concept required by the `MatrixCSR`
///        constructor.
///
/// Holds non-owning views into externally managed CSR arrays together with
/// the row and column `IndexMap`s that describe the parallel distribution.
/// It is intended as a short-lived helper: to be constructed it from the
/// output of `impl::matmul` and passed to the
/// `MatrixCSR` constructor.
///
/// @note All spans must remain valid for the lifetime of this object.
struct Sparsity
{
  /// @brief Row `IndexMap` — describes the parallel distribution of rows.
  std::shared_ptr<const common::IndexMap> _row_map;

  /// @brief Column `IndexMap` — describes the parallel distribution of
  ///        columns, including any ghost columns needed for the
  ///        off-diagonal block.
  std::shared_ptr<const common::IndexMap> _col_map;

  /// @brief CSR column indices, length `nnz`.  Values in
  ///        `[0, size_local(col) + num_ghosts(col))`, sorted within each row.
  std::span<const std::int32_t> _cols;

  /// @brief CSR row pointers, length `num_rows + 1`.  `_offsets[i]` is the
  ///        start of row `i` in `_cols`; `_offsets[num_rows]` equals `nnz`.
  std::span<const std::int64_t> _offsets;

  /// @brief Per-row diagonal block size, length `num_rows`.
  ///        `_off_diag[i]` is the number of entries in row `i` whose column
  ///        index is strictly less than `_col_map->size_local()`, i.e. the
  ///        count of entries in the diagonal (owned-column) block.  Entries
  ///        at positions `[_off_diag[i], row_end)` belong to the
  ///        off-diagonal block.
  std::span<const std::int32_t> _off_diag;

  /// @brief Return the row (`dim == 0`) or column (`dim == 1`) `IndexMap`.
  std::shared_ptr<const common::IndexMap> index_map(int dim) const
  {
    return dim == 0 ? _row_map : _col_map;
  }

  /// @brief Return the block size. Always 1 — this struct does not yet support
  ///        block-structured sparsity patterns.
  int block_size(int) const { return 1; }

  /// @brief Return the CSR graph as `(column_indices, row_pointers)`.
  std::pair<std::span<const std::int32_t>, std::span<const std::int64_t>>
  graph() const
  {
    return {_cols, _offsets};
  }

  /// @brief Return the per-row diagonal block sizes (see `_off_diag`).
  std::span<const std::int32_t> off_diagonal_offsets() const
  {
    return _off_diag;
  }
};

/// @brief Fetch the rows of Matrix B which are referenced by the ghost
/// columns of Matrix A.
/// @param A MatrixCSR
/// @param B MatrixCSR
/// @returns Tuple containing [new index map, rowptr, cols, values] for the received rows
template <typename T>
std::tuple<std::shared_ptr<common::IndexMap>, std::vector<std::int32_t>,
           std::vector<std::int32_t>, std::vector<T>>
fetch_ghost_rows(const dolfinx::la::MatrixCSR<T>& A,
                 const dolfinx::la::MatrixCSR<T>& B)
{
  // The ghost columns of A are global row indices into B.
  auto col_map_A = A.index_map(1); // column IndexMap of A
  auto row_map_B = B.index_map(0); // row IndexMap of B

  // Create neighborhood comms for col_map_A
  std::span<const int> src = col_map_A->src();
  std::span<const int> dest = col_map_A->dest();
  MPI_Comm comm = col_map_A->comm();

  // Serial fast-path: with a single rank there are no ghost columns and
  // no MPI communication to perform.
  // No-neighbour parallel fast-path: A has no ghost columns (src empty) and
  // no rank ghosts our A-owned columns (dest empty), so no remote rows of B
  // are needed and no rank is waiting for us to participate in a collective.
  // Symmetric topology guarantees the early return is safe without any global
  // synchronisation (same argument as for transpose()).
  // Use col_map_B — not col_map_A — for new_col_map so that any existing
  // ghost columns in B are preserved in the product's column map.

  int comm_size = dolfinx::MPI::size(col_map_A->comm());
  if (comm_size == 1 or (src.empty() && dest.empty()))
  {
    auto col_map_B = B.index_map(1);
    auto new_col_map = std::make_shared<dolfinx::common::IndexMap>(
        col_map_B->comm(), col_map_B->size_local(), col_map_B->ghosts(),
        col_map_B->owners());
    return {new_col_map, std::vector<std::int32_t>{0},
            std::vector<std::int32_t>{}, std::vector<T>{}};
  }

  // src is guaranteed sorted (Scatterer asserts this), so use binary search
  // rather than a heap-allocated map.
  auto rank_to_nbr = [&src](int r) -> int
  {
    auto it = std::lower_bound(src.begin(), src.end(), r);
    assert(it != src.end() && *it == r);
    return static_cast<int>(std::distance(src.begin(), it));
  };
  MPI_Comm neigh_comm_fwd;
  MPI_Comm neigh_comm_rev;
  MPI_Dist_graph_create_adjacent(comm, dest.size(), dest.data(), MPI_UNWEIGHTED,
                                 src.size(), src.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neigh_comm_fwd);
  MPI_Dist_graph_create_adjacent(comm, src.size(), src.data(), MPI_UNWEIGHTED,
                                 dest.size(), dest.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neigh_comm_rev);

  // Global indices of the rows of B we need (= ghost cols of A)
  std::span<const std::int64_t> required_globals_map = col_map_A->ghosts();

  // Owning rank for each ghost col of A (= source rank for each needed row)
  std::span<const int> ghost_owners = col_map_A->owners();

  std::vector<int> send_count(src.size(), 0);
  std::vector<int> recv_count(dest.size(), 0);

  for (int gh : ghost_owners)
    ++send_count[rank_to_nbr(gh)];
  MPI_Neighbor_alltoall(send_count.data(), 1, MPI_INT, recv_count.data(), 1,
                        MPI_INT, neigh_comm_fwd);

  // Send and recv displacements
  std::vector<int> send_disp(src.size() + 1, 0);
  std::partial_sum(send_count.begin(), send_count.end(),
                   std::next(send_disp.begin()));
  std::vector<int> recv_disp(dest.size() + 1, 0);
  std::partial_sum(recv_count.begin(), recv_count.end(),
                   std::next(recv_disp.begin()));

  // perm[i] = position in the send buffer (and received data) of original
  // ghost column i, so we can invert the ordering after receiving.
  std::vector<std::int32_t> perm(ghost_owners.size());
  std::vector<std::int64_t> required_globals(required_globals_map.size());
  for (std::size_t i = 0; i < ghost_owners.size(); ++i)
  {
    int pos = rank_to_nbr(ghost_owners[i]);
    perm[i] = send_disp[pos]; // position before increment
    required_globals[send_disp[pos]++] = required_globals_map[i];
  }

  // Reset send_disp
  send_disp[0] = 0;
  std::partial_sum(send_count.begin(), send_count.end(),
                   std::next(send_disp.begin()));

  // Send the global row indices we need; receive the indices others need from
  // us send buffer = required_globals (already ordered by src[])
  std::vector<std::int64_t> recv_row_globals(recv_disp.back());

  MPI_Neighbor_alltoallv(required_globals.data(), send_count.data(),
                         send_disp.data(), MPI_INT64_T, recv_row_globals.data(),
                         recv_count.data(), recv_disp.data(), MPI_INT64_T,
                         neigh_comm_fwd);

  // Convert received global row indices to local row indices in B.
  // The received indices are global rows of B that remote ranks need.
  // They must all be owned by this rank (by construction).
  std::vector<std::int32_t> recv_row_locals(recv_row_globals.size());
  row_map_B->global_to_local(recv_row_globals, recv_row_locals);

  // Compute size of send_data and communicate
  auto row_ptr_B = B.row_ptr();

  std::vector<int> send_entry_count(dest.size(), 0);
  std::vector<int> send_entry_disp(dest.size() + 1, 0);
  std::vector<int> recv_entry_count(src.size(), 0);
  std::vector<int> recv_entry_disp(src.size() + 1, 0);
  assert(recv_count.size() == dest.size());
  for (std::size_t p = 0; p < recv_count.size(); ++p)
  {
    for (std::int32_t i = recv_disp[p]; i < recv_disp[p + 1]; ++i)
    {
      int r = recv_row_locals[i];
      send_entry_count[p] += row_ptr_B[r + 1] - row_ptr_B[r];
    }
  }
  std::partial_sum(send_entry_count.begin(), send_entry_count.end(),
                   std::next(send_entry_disp.begin()));

  MPI_Neighbor_alltoall(send_entry_count.data(), 1, MPI_INT,
                        recv_entry_count.data(), 1, MPI_INT, neigh_comm_rev);

  std::partial_sum(recv_entry_count.begin(), recv_entry_count.end(),
                   std::next(recv_entry_disp.begin()));

  // Pack and send data values
  auto values_B = B.values();
  std::vector<T> send_vals(send_entry_disp.back());
  std::vector<T> recv_vals(recv_entry_disp.back());
  std::size_t k = 0;
  for (int r : recv_row_locals)
  {
    std::size_t row_size = row_ptr_B[r + 1] - row_ptr_B[r];
    std::copy(std::next(values_B.begin(), row_ptr_B[r]),
              std::next(values_B.begin(), row_ptr_B[r + 1]),
              std::next(send_vals.begin(), k));
    k += row_size;
  }
  MPI_Datatype mpi_T = dolfinx::MPI::mpi_t<T>;
  MPI_Neighbor_alltoallv(send_vals.data(), send_entry_count.data(),
                         send_entry_disp.data(), mpi_T, recv_vals.data(),
                         recv_entry_count.data(), recv_entry_disp.data(), mpi_T,
                         neigh_comm_rev);

  // Pack and send column indices
  auto cols_B = B.cols();
  auto col_map_B = B.index_map(1);
  const std::int32_t num_owned_cols_B = col_map_B->size_local();
  std::span<const int> ghost_owners_B = col_map_B->owners();

  // Send column indices (global indexing) and their owner ranks
  std::vector<std::int64_t> send_col_data(send_entry_disp.back());
  std::vector<std::int64_t> recv_col_data(recv_entry_disp.back());
  std::vector<int> send_col_owners(send_entry_disp.back());
  std::vector<int> recv_col_owners(recv_entry_disp.back());

  // Send row sizes so receivers can reconstruct the CSR row_ptr
  std::vector<int> send_row_size(recv_disp.back());
  std::vector<int> recv_row_size(send_disp.back());

  k = 0;
  int rank = dolfinx::MPI::rank(col_map_A->comm());
  for (std::size_t p = 0; p < dest.size(); ++p)
  {
    for (std::int32_t i = recv_disp[p]; i < recv_disp[p + 1]; ++i)
    {
      int r = recv_row_locals[i];
      std::size_t row_size = row_ptr_B[r + 1] - row_ptr_B[r];
      col_map_B->local_to_global(
          std::span(std::next(cols_B.begin(), row_ptr_B[r]), row_size),
          std::span(std::next(send_col_data.begin(), k), row_size));
      for (std::size_t j = 0; j < row_size; ++j)
      {
        std::int32_t local_col = cols_B[row_ptr_B[r] + j];
        send_col_owners[k + j]
            = (local_col < num_owned_cols_B)
                  ? rank
                  : ghost_owners_B[local_col - num_owned_cols_B];
      }
      k += row_size;
      send_row_size[i] = row_size;
    }
  }

  MPI_Neighbor_alltoallv(send_col_data.data(), send_entry_count.data(),
                         send_entry_disp.data(), MPI_INT64_T,
                         recv_col_data.data(), recv_entry_count.data(),
                         recv_entry_disp.data(), MPI_INT64_T, neigh_comm_rev);
  MPI_Neighbor_alltoallv(send_col_owners.data(), send_entry_count.data(),
                         send_entry_disp.data(), MPI_INT,
                         recv_col_owners.data(), recv_entry_count.data(),
                         recv_entry_disp.data(), MPI_INT, neigh_comm_rev);
  MPI_Neighbor_alltoallv(send_row_size.data(), recv_count.data(),
                         recv_disp.data(), MPI_INT, recv_row_size.data(),
                         send_count.data(), send_disp.data(), MPI_INT,
                         neigh_comm_rev);

  MPI_Comm_free(&neigh_comm_fwd);
  MPI_Comm_free(&neigh_comm_rev);

  // Build CSR row pointer for the received ghost rows.
  // recv_vals and recv_col_data are the values and (global) column indices;
  // recv_row_size[i] is the number of entries in ghost row i.
  std::vector<std::int32_t> ghost_row_ptr(recv_row_size.size() + 1, 0);
  std::partial_sum(recv_row_size.begin(), recv_row_size.end(),
                   std::next(ghost_row_ptr.begin()));

  // ghost_row_ptr, recv_vals, recv_col_data now form a CSR structure for
  // the ghost rows of B needed by this rank.

  // Collect ghost column indices with their owners.
  // Start from the received ghost rows, then add existing ghosts of col_map_B.
  // Exclude anything in the locally owned range of col_map_B, then deduplicate.
  auto [col_B_local_start, col_B_local_end] = col_map_B->local_range();
  auto is_local = [col_B_local_start, col_B_local_end](std::int64_t idx)
  { return idx >= col_B_local_start && idx < col_B_local_end; };

  // Merge received (global_col, owner) pairs with the existing ghosts.
  std::vector<std::pair<std::int64_t, int>> col_owner_pairs;
  col_owner_pairs.reserve(recv_col_data.size() + col_map_B->ghosts().size());
  for (std::size_t i = 0; i < recv_col_data.size(); ++i)
    if (!is_local(recv_col_data[i]))
      col_owner_pairs.push_back({recv_col_data[i], recv_col_owners[i]});

  std::span<const std::int64_t> existing_ghosts = col_map_B->ghosts();
  std::span<const int> existing_ghost_owners = col_map_B->owners();
  for (std::size_t i = 0; i < existing_ghosts.size(); ++i)
    col_owner_pairs.push_back({existing_ghosts[i], existing_ghost_owners[i]});

  // Sort by global index and deduplicate.
  std::sort(col_owner_pairs.begin(), col_owner_pairs.end());
  col_owner_pairs.erase(
      std::unique(col_owner_pairs.begin(), col_owner_pairs.end()),
      col_owner_pairs.end());

  std::vector<std::int64_t> unique_cols(col_owner_pairs.size());
  std::vector<int> unique_col_owners(col_owner_pairs.size());
  for (std::size_t i = 0; i < col_owner_pairs.size(); ++i)
  {
    unique_cols[i] = col_owner_pairs[i].first;
    unique_col_owners[i] = col_owner_pairs[i].second;
  }

  // Build the extended column IndexMap for B.
  auto new_col_map = std::make_shared<dolfinx::common::IndexMap>(
      comm, col_map_B->size_local(), unique_cols, unique_col_owners);

  // Convert received global column indices to local indices using new_col_map.
  std::vector<std::int32_t> recv_col_local(recv_col_data.size());
  new_col_map->global_to_local(recv_col_data, recv_col_local);

  // The received data is in send-buffer order (grouped by src rank).
  // Reorder it back to the original ghost column order of col_map_A so that
  // sparsity_pattern and matmul can index by (j - num_local_rows_B).
  const std::size_t num_ghosts = perm.size();
  std::vector<std::int32_t> orig_ghost_row_ptr(num_ghosts + 1, 0);
  for (std::size_t i = 0; i < num_ghosts; ++i)
    orig_ghost_row_ptr[i + 1] = recv_row_size[perm[i]];
  std::partial_sum(orig_ghost_row_ptr.begin(), orig_ghost_row_ptr.end(),
                   orig_ghost_row_ptr.begin());

  std::vector<std::int32_t> orig_recv_col_local(recv_col_local.size());
  std::vector<T> orig_recv_vals(recv_vals.size());
  for (std::size_t i = 0; i < num_ghosts; ++i)
  {
    std::int32_t src_start = ghost_row_ptr[perm[i]];
    std::int32_t src_end = ghost_row_ptr[perm[i] + 1];
    std::int32_t dst_start = orig_ghost_row_ptr[i];
    std::copy(recv_col_local.begin() + src_start,
              recv_col_local.begin() + src_end,
              orig_recv_col_local.begin() + dst_start);
    std::copy(recv_vals.begin() + src_start, recv_vals.begin() + src_end,
              orig_recv_vals.begin() + dst_start);
  }

  return {new_col_map, orig_ghost_row_ptr, orig_recv_col_local, orig_recv_vals};
}

/// @brief Compute the sparsity pattern and values of C = A*B in a single pass.
///
/// Uses a dense accumulator (one entry per possible output column) to
/// simultaneously detect nonzero columns and accumulate A[i,j]*B[j,k],
/// eliminating the separate value-fill loop and the per-entry lower_bound
/// search that a two-pass approach requires.
///
/// @param A             Left matrix.
/// @param B             Right matrix (local rows only).
/// @param new_col_map   Extended column IndexMap for C, from fetch_ghost_rows.
/// @param ghost_row_ptr CSR row pointer for the ghost rows of B.
/// @param ghost_cols    Local column indices (w.r.t. new_col_map) for ghost
/// rows.
/// @param ghost_vals    Values for the ghost rows of B.
/// @return Tuple (row_ptr, off_diag_offsets, cols, vals). off_diag_offsets[i]
///         is the number of diagonal-block entries in row i, computed during
///         the same sort step that establishes column order.
template <typename T>
std::tuple<std::vector<std::int64_t>, std::vector<std::int32_t>,
           std::vector<std::int32_t>, std::vector<T>>
matmul(const dolfinx::la::MatrixCSR<T>& A, const dolfinx::la::MatrixCSR<T>& B,
       std::shared_ptr<const common::IndexMap> new_col_map,
       std::span<const std::int32_t> ghost_row_ptr,
       std::span<const std::int32_t> ghost_cols, std::span<const T> ghost_vals)
{
  auto col_map_B = B.index_map(1);
  const std::int32_t num_local_rows_A = A.index_map(0)->size_local();
  const std::int32_t num_local_rows_B = B.index_map(0)->size_local();
  const std::int32_t num_owned_cols_B = col_map_B->size_local();
  const std::int32_t num_owned_cols_C = new_col_map->size_local();

  auto row_ptr_A = A.row_ptr();
  auto offdiag_ptr_A = A.off_diag_offset();
  auto cols_A = A.cols();
  auto vals_A = A.values();
  auto row_ptr_B = B.row_ptr();
  auto cols_B = B.cols();
  auto vals_B = B.values();

  // Remap B's ghost column indices into new_col_map's index space.
  // Owned column indices (0..num_owned_cols_B-1) are identical in both maps.
  std::span<const std::int64_t> b_ghost_globals = col_map_B->ghosts();
  std::vector<std::int32_t> b_ghost_remap(b_ghost_globals.size());
  new_col_map->global_to_local(b_ghost_globals, b_ghost_remap);

  const std::int32_t num_cols_C
      = new_col_map->size_local()
        + static_cast<std::int32_t>(new_col_map->ghosts().size());

  std::vector<std::int64_t> row_ptr_C = {0};
  row_ptr_C.reserve(num_local_rows_A + 1);
  std::vector<std::int32_t> off_diag_offsets_C;
  off_diag_offsets_C.reserve(num_local_rows_A);
  std::vector<std::int32_t> cols_C;
  std::vector<T> vals_C;

  // Dense accumulator: acc[k] holds the running sum for column k.
  // in_row[k] marks whether column k was touched in the current row.
  // Both are reset via row_cols at the end of each row (O(nnz/row), not
  // O(num_cols_C)).
  std::vector<T> acc(num_cols_C, T(0));
  std::vector<bool> in_row(num_cols_C, false);
  std::vector<std::int32_t> row_cols;

  for (std::int32_t i = 0; i < num_local_rows_A; ++i)
  {
    row_cols.clear();

    // Local columns of A  →  local rows of B.
    for (std::int32_t ka = row_ptr_A[i]; ka < offdiag_ptr_A[i]; ++ka)
    {
      const std::int32_t j = cols_A[ka];
      const T a = vals_A[ka];
      for (std::int32_t kb = row_ptr_B[j]; kb < row_ptr_B[j + 1]; ++kb)
      {
        const std::int32_t c = cols_B[kb];
        const std::int32_t k
            = c < num_owned_cols_B ? c : b_ghost_remap[c - num_owned_cols_B];
        if (!in_row[k])
        {
          in_row[k] = true;
          row_cols.push_back(k);
        }
        acc[k] += a * vals_B[kb];
      }
    }

    // Ghost columns of A  →  ghost rows of B.
    for (std::int32_t ka = offdiag_ptr_A[i]; ka < row_ptr_A[i + 1]; ++ka)
    {
      const std::int32_t j = cols_A[ka];
      const T a = vals_A[ka];
      const std::int32_t g = j - num_local_rows_B;
      for (std::int32_t kb = ghost_row_ptr[g]; kb < ghost_row_ptr[g + 1]; ++kb)
      {
        const std::int32_t k = ghost_cols[kb];
        if (!in_row[k])
        {
          in_row[k] = true;
          row_cols.push_back(k);
        }
        acc[k] += a * ghost_vals[kb];
      }
    }

    // Sort touched indices to satisfy the CSR sorted-column invariant
    // required by off_diag_offset and MatrixCSR.
    std::sort(row_cols.begin(), row_cols.end());

    // Diagonal block boundary: count entries in the owned column range.
    // row_cols is sorted, so a single lower_bound suffices.
    off_diag_offsets_C.push_back(static_cast<std::int32_t>(
        std::lower_bound(row_cols.begin(), row_cols.end(), num_owned_cols_C)
        - row_cols.begin()));

    // Flush accumulator to output and reset both acc and in_row.
    for (const std::int32_t c : row_cols)
    {
      cols_C.push_back(c);
      vals_C.push_back(acc[c]);
      acc[c] = T(0);
      in_row[c] = false;
    }
    row_ptr_C.push_back(static_cast<std::int64_t>(cols_C.size()));
  }

  return {std::move(row_ptr_C), std::move(off_diag_offsets_C),
          std::move(cols_C), std::move(vals_C)};
}

} // namespace impl

/// @brief Compute C = A*B as a distributed MatrixCSR.
///
/// @param A Left matrix.
/// @param B Right matrix.
/// @note Currently only supports block-size 1 matrices.
/// @return The product C = A*B as a MatrixCSR with row distribution
///         matching A and column distribution determined by B.
///         The row IndexMap of C has no ghosts, and the column IndexMap of C
///         will generally be a larger superset of the column IndexMap of B.
template <typename T>
dolfinx::la::MatrixCSR<T> matmul(const dolfinx::la::MatrixCSR<T>& A,
                                 const dolfinx::la::MatrixCSR<T>& B)
{
  dolfinx::common::Timer t_spgemm("MatrixCSR SpGEMM");

  // Fetch ghost rows of B needed to multiply against A's ghost columns.
  spdlog::debug("Fetch remote rows of B in C=A*B");
  auto [new_col_map, ghost_row_ptr, ghost_cols, ghost_vals]
      = impl::fetch_ghost_rows(A, B);

  // Single pass: compute sparsity, off-diagonal boundaries, and values.
  spdlog::debug("Compute sparsity and values of C=A*B");
  auto [C_row_ptr, C_off_diag_offsets, C_cols, C_vals_vec] = impl::matmul(
      A, B, new_col_map, std::span<const std::int32_t>(ghost_row_ptr),
      std::span<const std::int32_t>(ghost_cols),
      std::span<const T>(ghost_vals));

  auto C_row_map = std::make_shared<common::IndexMap>(
      A.index_map(0)->comm(), A.index_map(0)->size_local());
  impl::Sparsity sp{C_row_map, new_col_map, C_cols, C_row_ptr,
                    C_off_diag_offsets};
  dolfinx::la::MatrixCSR<T> C(sp);
  std::copy(C_vals_vec.begin(), C_vals_vec.end(), C.values().begin());

  return C;
}

} // namespace dolfinx::la
