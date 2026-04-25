// Copyright (C) 2026 Chris Richardson
// SPDX-License-Identifier: MIT

#pragma once

#include <dolfinx/common/IndexMap.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/matmul.h>
#include <mpi.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <map>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace dolfinx::la
{
namespace impl
{
/// @brief Compute the local (diagonal-block) transpose of A.
///
/// Iterates over owned rows of A and the owned-column entries within each
/// row (indices [row_ptr[i], off_diag_offset[i])), building the transpose
/// CSR in one bucket-fill pass.  The resulting column indices are original
/// row indices (0-based local), so columns within every output row are
/// already in ascending order.
///
/// @param A MatrixCSR
/// @return Tuple (cols, row_ptr, values) for the transposed diagonal block.
template <typename T, int BS0 = -1, int BS1 = -1>
std::tuple<std::vector<std::int32_t>, std::vector<std::int64_t>, std::vector<T>>
local_transpose(const dolfinx::la::MatrixCSR<T>& A)
{
  spdlog::debug("local transpose");
  std::array<int, 2> bs = A.block_size();
  auto A_row_map = A.index_map(0);
  auto A_col_map = A.index_map(1);
  const auto A_cols = A.cols();
  const auto A_row_start = A.row_ptr();
  const auto A_row_end = A.off_diag_offset(); // end of owned-col block
  const auto A_vals = A.values();

  const std::int32_t n_rows = A_row_map->size_local();
  const std::int32_t n_cols = A_col_map->size_local(); // rows in A^T

  // Count entries per output row.
  std::vector<std::int64_t> row_count(n_cols, 0);
  for (std::int32_t i = 0; i < n_rows; ++i)
    for (std::int32_t k = A_row_start[i]; k < A_row_end[i]; ++k)
      ++row_count[A_cols[k]];

  // Exclusive prefix-sum → row pointer (used as write cursor below).
  std::vector<std::int64_t> row_offsetT(n_cols + 1, 0);
  std::partial_sum(row_count.begin(), row_count.end(),
                   std::next(row_offsetT.begin()));

  // Fill: column index in A^T is the original row index i.
  const std::int32_t total_nnz = row_offsetT.back();
  std::vector<std::int32_t> colsT(total_nnz);
  std::vector<T> valsT(total_nnz * bs[0] * bs[1]);
  std::vector<std::int64_t> cursor = row_offsetT;
  for (std::int32_t i = 0; i < n_rows; ++i)
  {
    for (std::int32_t k = A_row_start[i]; k < A_row_end[i]; ++k)
    {
      const std::int32_t col = A_cols[k];
      const std::int32_t pos = cursor[col]++;
      colsT[pos] = i;
      if constexpr (BS0 < 0 or BS1 < 0)
      {
        for (int k0 = 0; k0 < bs[0]; ++k0)
          for (int k1 = 0; k1 < bs[1]; ++k1)
            valsT[pos * bs[0] * bs[1] + k0 * bs[1] + k1]
                = A_vals[k * bs[0] * bs[1] + k1 * bs[0] + k0];
      }
      else
      {
        for (int k0 = 0; k0 < BS0; ++k0)
          for (int k1 = 0; k1 < BS1; ++k1)
            valsT[pos * BS0 * BS1 + k0 * BS1 + k1]
                = A_vals[k * BS0 * BS1 + k1 * BS0 + k0];
      }
    }
  }

  return {std::move(colsT), std::move(row_offsetT), std::move(valsT)};
}
} // namespace impl

/// @brief Compute the distributed transpose of a MatrixCSR.
///
/// Given A with row map R (n_row owned rows) and column map C (n_col owned
/// columns plus any ghost columns from remote ranks), returns Aᵀ with:
///   row map  = ghost-free column map of A  (n_col owned rows, no ghosts)
///   col map  = row map of A extended with any ghost row indices that
///              appear as nonzeros contributed by remote ranks.
///
/// Communication strategy:
///   1. Exchange per-neighbour entry counts (MPI_Neighbor_alltoall).
///   2. Build displacements and pack send buffers.
///   3. Post non-blocking data exchange (three MPI_Ineighbor_alltoallv
///      in flight simultaneously: row global indices, column global
///      indices, and values).
///   4. While data exchange is in flight, compute the local
///      (diagonal-block) part of the transpose via local_transpose.
///   5. Wait for data, merge received entries, finalise sparsity and
///      construct the MatrixCSR.
///
/// @param A  Input matrix.
/// @return   Aᵀ as a distributed MatrixCSR.
template <typename T, int BS0 = -1, int BS1 = -1>
dolfinx::la::MatrixCSR<T> transpose(const dolfinx::la::MatrixCSR<T>& A)
{
  dolfinx::common::Timer tt("Transpose MatrixCSR");

  std::array<int, 2> bs = A.block_size();
  if constexpr (BS0 != -1 or BS1 != -1)
  {
    if (bs[0] != BS0 or bs[1] != BS1)
      throw std::runtime_error("Invalid template parameters/block size");
  }

  auto row_map_A = A.index_map(0); // row map
  auto col_map_A = A.index_map(1); // column map

  const std::int32_t n_row = row_map_A->size_local();
  const std::int32_t n_col = col_map_A->size_local();

  // Serial fast-path: no ghost columns, no communication needed.
  // (Same reasoning as in fetch_ghost_rows — must not be applied per-rank
  // in a parallel run, only when the communicator has a single process.)

  // col_map_A communication topology:
  //   src  = ranks that own our ghost columns → we SEND to them
  //   dest = ranks that ghost our owned columns → we RECEIVE from them
  std::span<const int> src = col_map_A->src();
  std::span<const int> dest = col_map_A->dest();

  // No-neighbour fast-path (parallel, but A's column map has no cross-process
  // topology — i.e. A has no ghost columns).
  //
  // When src AND dest are both empty this rank does not appear in any other
  // rank's neighbourhood communicator, so returning early without calling
  // neighbourhood collectives is safe: no other rank is blocked waiting for
  // us.
  //
  // We must NOT generalise this check to "dest is empty" alone: if src is
  // non-empty we have ghost columns to report to remote owners, meaning those
  // owners ARE expecting our participation in the alltoall calls.

  MPI_Comm comm = row_map_A->comm();
  int comm_size = 1;
  MPI_Comm_size(comm, &comm_size);
  if (comm_size == 1 or (src.empty() && dest.empty()))
  {
    // Pure local transpose: all columns are owned, so Aᵀ has no ghost columns.
    auto at_col_map = std::make_shared<dolfinx::common::IndexMap>(comm, n_row);
    auto at_row_map = std::make_shared<dolfinx::common::IndexMap>(comm, n_col);

    auto [at_cols, at_rp, at_vals] = impl::local_transpose<T, BS0, BS1>(A);

    // All columns are owned (< n_row) → off_diag_offset equals row width.
    std::vector<std::int32_t> off_diag_offsets(n_col);
    for (int j = 0; j < n_col; ++j)
      off_diag_offsets[j] = at_rp[j + 1] - at_rp[j];

    la::impl::Sparsity sp{at_row_map, at_col_map,       at_cols,
                          at_rp,      off_diag_offsets, {bs[1], bs[0]}};
    dolfinx::la::MatrixCSR<T> At(sp);
    std::ranges::copy(at_vals, At.values().begin());
    return At;
  }

  auto row_ptr_A = A.row_ptr();
  auto offset_row_A = A.off_diag_offset();
  auto cols_A = A.cols();
  auto vals_A = A.values();

  std::span<const std::int64_t> ghost_col_globals = col_map_A->ghosts();

  // Index into src[] giving the neighbor owning rank of a ghost column.
  std::vector<int> ghost_col_owners(col_map_A->owners().size());
  {
    std::map<int, int> owner_to_src_idx;
    for (std::size_t k = 0; k < src.size(); ++k)
      owner_to_src_idx[src[k]] = static_cast<int>(k);
    std::transform(col_map_A->owners().begin(), col_map_A->owners().end(),
                   ghost_col_owners.begin(), [&owner_to_src_idx](int i)
                   { return owner_to_src_idx.at(i); });
  }

  // Neighbourhood communicator: this rank sends to src[], receives from
  // dest[].
  MPI_Comm neigh_comm;
  MPI_Dist_graph_create_adjacent(
      comm, static_cast<int>(dest.size()), dest.data(), MPI_UNWEIGHTED,
      static_cast<int>(src.size()), src.data(), MPI_UNWEIGHTED, MPI_INFO_NULL,
      false, &neigh_comm);

  // -----------------------------------------------------------------------
  // Count off-process entries: A[i, j_ghost] grouped by owning rank.
  // -----------------------------------------------------------------------
  std::vector<int> send_count(src.size(), 0);
  for (std::int32_t i = 0; i < n_row; ++i)
  {
    for (std::int64_t k = offset_row_A[i]; k < row_ptr_A[i + 1]; ++k)
    {
      const std::int32_t j = cols_A[k] - n_col;
      ++send_count[ghost_col_owners[j]];
    }
  }

  spdlog::debug("Send data MPI");
  std::vector<int> recv_count(dest.size(), 0);
  MPI_Neighbor_alltoall(send_count.data(), 1, MPI_INT, recv_count.data(), 1,
                        MPI_INT, neigh_comm);

  std::vector<int> send_disp(src.size() + 1, 0);
  std::partial_sum(send_count.begin(), send_count.end(),
                   std::next(send_disp.begin()));
  std::vector<int> recv_disp(dest.size() + 1, 0);
  std::partial_sum(recv_count.begin(), recv_count.end(),
                   std::next(recv_disp.begin()));
  const int total_send = send_disp.back();
  const int total_recv = recv_disp.back();

  // -----------------------------------------------------------------------
  // For each A[i, j] with j < n_col (owned column), store the pair
  // (global row index, value) in local_cols[j].
  // -----------------------------------------------------------------------
  auto [row_start, row_end] = row_map_A->local_range();

  // Each send entry carries three values:
  //   send_row_gidx : global row index of A  (= column of Aᵀ on receiver)
  //   send_col_gidx : global column index of A  (= row of Aᵀ on receiver)
  //   send_vals     : matrix entry value
  std::vector<std::int64_t> send_row_gidx(total_send);
  std::vector<std::int64_t> send_col_gidx(total_send);
  std::vector<T> send_vals(total_send * bs[0] * bs[1]);
  {
    std::vector<int> pos = send_disp; // per-rank write cursor
    for (std::int32_t i = 0; i < n_row; ++i)
    {
      const std::int64_t global_i = row_start + i;
      for (std::int64_t k = offset_row_A[i]; k < row_ptr_A[i + 1]; ++k)
      {
        const std::int32_t j = cols_A[k] - n_col;
        const int sidx = ghost_col_owners[j];
        const int p = pos[sidx]++;
        send_row_gidx[p] = global_i;
        send_col_gidx[p] = ghost_col_globals[j];
        if constexpr (BS0 < 0 or BS1 < 0)
        {
          for (int k0 = 0; k0 < bs[0]; ++k0)
            for (int k1 = 0; k1 < bs[1]; ++k1)
              send_vals[p * bs[0] * bs[1] + k0 * bs[1] + k1]
                  = vals_A[k * bs[0] * bs[1] + k0 * bs[1] + k1];
        }
        else
        {
          for (int k0 = 0; k0 < BS0; ++k0)
            for (int k1 = 0; k1 < BS1; ++k1)
              send_vals[p * BS0 * BS1 + k0 * BS1 + k1]
                  = vals_A[k * BS0 * BS1 + k0 * BS1 + k1];
        }
      }
    }
  }

  // -----------------------------------------------------------------------
  // POST non-blocking data exchange (three messages simultaneously).
  // -----------------------------------------------------------------------
  std::vector<std::int64_t> recv_row_gidx(total_recv);
  std::vector<std::int64_t> recv_col_gidx(total_recv);
  std::vector<T> recv_vals(total_recv * bs[0] * bs[1]);

  MPI_Request data_reqs[3];
  MPI_Datatype mpi_T;
  MPI_Type_contiguous(bs[0] * bs[1], dolfinx::MPI::mpi_t<T>, &mpi_T);
  MPI_Type_commit(&mpi_T);

  MPI_Ineighbor_alltoallv(send_row_gidx.data(), send_count.data(),
                          send_disp.data(), MPI_INT64_T, recv_row_gidx.data(),
                          recv_count.data(), recv_disp.data(), MPI_INT64_T,
                          neigh_comm, &data_reqs[0]);
  MPI_Ineighbor_alltoallv(send_col_gidx.data(), send_count.data(),
                          send_disp.data(), MPI_INT64_T, recv_col_gidx.data(),
                          recv_count.data(), recv_disp.data(), MPI_INT64_T,
                          neigh_comm, &data_reqs[1]);
  MPI_Ineighbor_alltoallv(send_vals.data(), send_count.data(), send_disp.data(),
                          mpi_T, recv_vals.data(), recv_count.data(),
                          recv_disp.data(), mpi_T, neigh_comm, &data_reqs[2]);

  // Do local part
  auto [at0_cols, at0_rp, at0_vals] = impl::local_transpose<T, BS0, BS1>(A);

  // -----------------------------------------------------------------------
  // Wait for all data exchanges to complete.
  // -----------------------------------------------------------------------
  MPI_Waitall(3, data_reqs, MPI_STATUSES_IGNORE);
  MPI_Type_free(&mpi_T);
  MPI_Comm_free(&neigh_comm);
  spdlog::debug("got values MPI");

  // -----------------------------------------------------------------------
  // Merge received entries into local_cols and collect ghost row info.
  // -----------------------------------------------------------------------
  auto [col_start, col_end] = col_map_A->local_range();

  // Count number of entries on each row of Aᵀ and build offset
  std::vector<std::int64_t> new_row_count(n_col);
  for (std::size_t i = 0; i < n_col; ++i)
    new_row_count[i] = static_cast<int>(at0_rp[i + 1] - at0_rp[i]);
  for (std::int64_t c : recv_col_gidx)
  {
    const std::int32_t local_col = static_cast<std::int32_t>(c - col_start);
    assert(local_col >= 0 && local_col < n_col);
    ++new_row_count[local_col];
  }
  std::vector<std::int64_t> at_row_ptr(new_row_count.size() + 1, 0);
  std::partial_sum(new_row_count.begin(), new_row_count.end(),
                   std::next(at_row_ptr.begin()));

  // Compute new IndexMap for columns
  std::vector<std::int64_t> ghost_row_globals;
  std::vector<int> ghost_row_owners;
  {
    // ghost_row_pairs: (global row index, owner rank) for Aᵀ column map.
    std::vector<std::pair<std::int64_t, int>> ghost_row_pairs;
    ghost_row_pairs.reserve(recv_disp.back());
    for (std::size_t p = 0; p < dest.size(); ++p)
    {
      for (int k = recv_disp[p]; k < recv_disp[p + 1]; ++k)
      {
        const std::int64_t gidx = recv_row_gidx[k];
        assert(gidx < row_start || gidx >= row_end);
        ghost_row_pairs.emplace_back(gidx, static_cast<int>(dest[p]));
      }
    }

    // -----------------------------------------------------------------------
    // Build Aᵀ column map (n_row owned + received ghosts).
    // -----------------------------------------------------------------------
    std::sort(ghost_row_pairs.begin(), ghost_row_pairs.end());
    ghost_row_pairs.erase(
        std::unique(ghost_row_pairs.begin(), ghost_row_pairs.end()),
        ghost_row_pairs.end());

    ghost_row_globals.reserve(ghost_row_pairs.size());
    ghost_row_owners.reserve(ghost_row_pairs.size());
    for (auto& [g, o] : ghost_row_pairs)
    {
      ghost_row_globals.push_back(g);
      ghost_row_owners.push_back(o);
    }
  }

  auto at_col_map = std::make_shared<dolfinx::common::IndexMap>(
      comm, n_row, ghost_row_globals, ghost_row_owners);

  // Batch convert received rows (cols of Aᵀ) into local indexing
  std::vector<std::int32_t> new_col_local(recv_disp.back());
  at_col_map->global_to_local(recv_row_gidx, new_col_local);

  // Copy over local rows into new data structure
  std::vector<std::int32_t> at_cols(at_row_ptr.back());
  std::vector<T> at_vals(at_row_ptr.back() * bs[0] * bs[1]);
  for (std::size_t i = 0; i < n_col; ++i)
  {
    std::copy(std::next(at0_cols.begin(), at0_rp[i]),
              std::next(at0_cols.begin(), at0_rp[i + 1]),
              std::next(at_cols.begin(), at_row_ptr[i]));
    std::copy(std::next(at0_vals.begin(), at0_rp[i] * bs[0] * bs[1]),
              std::next(at0_vals.begin(), at0_rp[i + 1] * bs[0] * bs[1]),
              std::next(at_vals.begin(), at_row_ptr[i] * bs[0] * bs[1]));
  }

  // Append received values.
  // cursor[j] = next free write position in at_cols/at_vals for row j,
  // initialised to just after the local entries already copied in.
  std::vector<std::int32_t> cursor(n_col);
  std::vector<std::int32_t> off_diag_offsets(n_col);
  for (int j = 0; j < n_col; ++j)
  {
    off_diag_offsets[j] = at0_rp[j + 1] - at0_rp[j];
    cursor[j] = static_cast<std::int32_t>(at_row_ptr[j]) + off_diag_offsets[j];
  }

  for (std::size_t p = 0; p < dest.size(); ++p)
  {
    for (int k = recv_disp[p]; k < recv_disp[p + 1]; ++k)
    {
      const std::int32_t local_col
          = static_cast<std::int32_t>(recv_col_gidx[k] - col_start);
      assert(local_col >= 0 && local_col < n_col);
      const int pos = cursor[local_col]++;
      at_cols[pos] = new_col_local[k];
      if constexpr (BS0 < 0 or BS1 < 0)
      {
        for (int k0 = 0; k0 < bs[0]; ++k0)
          for (int k1 = 0; k1 < bs[1]; ++k1)
            at_vals[pos * bs[0] * bs[1] + k0 * bs[1] + k1]
                = recv_vals[k * bs[0] * bs[1] + bs[0] * k1 + k0];
      }
      else
      {
        for (int k0 = 0; k0 < BS0; ++k0)
          for (int k1 = 0; k1 < BS1; ++k1)
            at_vals[pos * BS0 * BS1 + k0 * BS1 + k1]
                = recv_vals[k * BS0 * BS1 + BS0 * k1 + k0];
      }
    }
  }

  // -----------------------------------------------------------------------
  // Construct Aᵀ as MatrixCSR.
  // Row map: ghost-free (n_col owned rows, no ghost rows needed).
  // -----------------------------------------------------------------------
  auto at_row_map = std::make_shared<dolfinx::common::IndexMap>(comm, n_col);

  la::impl::Sparsity sp{at_row_map, at_col_map,       at_cols,
                        at_row_ptr, off_diag_offsets, {bs[1], bs[0]}};

  dolfinx::la::MatrixCSR<T> At(sp);
  std::ranges::copy(at_vals, At.values().begin());

  spdlog::info("Transpose: ({} x {}) -> ({} x {}), nnz {}",
               row_map_A->size_global(), col_map_A->size_global(),
               col_map_A->size_global(), row_map_A->size_global(),
               at_row_ptr.back());

  return At;
}

} // namespace dolfinx::la
