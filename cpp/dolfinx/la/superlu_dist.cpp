// Copyright (C) 2026 Jack S. Hale, Chris N. Richardson
// Copyright (C) 2026 Garth N. Wells, Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_SUPERLU_DIST

#include "superlu_dist.h"
extern "C"
{
#include <superlu_ddefs.h>
#include <superlu_sdefs.h>
#include <superlu_zdefs.h>
}
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>
#include <format>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <mpi.h>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::la;

// Trick for declaring anonymous typedef structs from SuperLU_DIST
struct dolfinx::la::SuperLUDistStructs::SuperMatrix : public ::SuperMatrix
{
};
//----------------------------------------------------------------------------
/// Struct holding vector of type int_t
struct dolfinx::la::SuperLUDistStructs::vec_int_t
{
  /// @brief vector
  std::vector<int_t> vec;
};
//----------------------------------------------------------------------------
void SuperMatrixDeleter::operator()(
    SuperLUDistStructs::SuperMatrix* supermatrix) const noexcept
{
  Destroy_SuperMatrix_Store_dist(supermatrix);
  delete supermatrix;
}
//----------------------------------------------------------------------------
namespace
{
/// @brief Check that a size or index fits in SuperLU_DIST's `int_t`.
/// @param[in] x Size or index.
/// @throws std::out_of_range if `x` exceeds the range of `int_t`.
void check_int_t(std::int64_t x)
{
  if (x >= static_cast<std::int64_t>(std::numeric_limits<int_t>::max()))
    throw std::out_of_range("Value outside upper range of SuperLU_DIST int_t.");
}
//----------------------------------------------------------------------------
/// @brief Create a SuperMatrix from the local rows of a distributed
/// matrix.
///
/// Sizes and indices are of unrolled (scalar) rows and columns.
///
/// @param[in] m Global number of rows.
/// @param[in] n Global number of columns.
/// @param[in] m_loc Number of local rows.
/// @param[in] first_row Global index of the first local row.
/// @param[in] A_mat_values Values of the local rows.
/// @param[in] rowptr Row pointer of the local rows.
/// @param[in] cols Global column indices of the local rows.
/// @return The SuperMatrix, which references the arrays.
/// @throws std::invalid_argument if `m != n`.
/// @throws std::out_of_range if a size or index exceeds `int_t`.
template <typename T>
std::unique_ptr<SuperLUDistStructs::SuperMatrix, SuperMatrixDeleter>
create_supermatrix(std::int64_t m, std::int64_t n, std::int32_t m_loc,
                   std::int64_t first_row, auto& A_mat_values, auto& rowptr,
                   auto& cols)
{
  logger().info("Start create_supermatrix");

  if (m != n)
    throw std::invalid_argument("Cannot solve non-square system");

  // Local number of non-zeros, checked by the caller before `rowptr`
  // was built in `int_t`
  std::int64_t nnz_loc = rowptr.vec.at(m_loc);

  check_int_t(m);
  check_int_t(n);
  check_int_t(m_loc);
  check_int_t(first_row);
  check_int_t(nnz_loc);

  std::unique_ptr<SuperLUDistStructs::SuperMatrix, SuperMatrixDeleter> p(
      new SuperLUDistStructs::SuperMatrix, SuperMatrixDeleter{});

  if constexpr (std::is_same_v<T, double>)
  {
    dCreate_CompRowLoc_Matrix_dist(
        p.get(), m, n, nnz_loc, m_loc, first_row, A_mat_values.data(),
        cols.vec.data(), rowptr.vec.data(), SLU_NR_loc, SLU_D, SLU_GE);
  }
  else if constexpr (std::is_same_v<T, float>)
  {
    sCreate_CompRowLoc_Matrix_dist(
        p.get(), m, n, nnz_loc, m_loc, first_row, A_mat_values.data(),
        cols.vec.data(), rowptr.vec.data(), SLU_NR_loc, SLU_S, SLU_GE);
  }
  else if constexpr (std::is_same_v<T, std::complex<double>>)
  {
    zCreate_CompRowLoc_Matrix_dist(
        p.get(), m, n, nnz_loc, m_loc, first_row,
        reinterpret_cast<doublecomplex*>(A_mat_values.data()), cols.vec.data(),
        rowptr.vec.data(), SLU_NR_loc, SLU_Z, SLU_GE);
  }
  else
    static_assert(dolfinx::la::impl::always_false_v<T>, "Invalid scalar type");

  logger().info("Finished create_supermatrix");
  return p;
}
//----------------------------------------------------------------------------
/// Blocks `A[i][j]` of a block matrix, `nullptr` for a zero block.
template <typename T>
using Blocks = std::span<const std::vector<const MatrixCSR<T>*>>;

/// Owned range `[first, last)` and global number of the unrolled rows
/// or columns of a matrix, or of a block row or block column.
using Layout = std::array<std::int64_t, 3>;

/// @brief Layout of the unrolled rows or columns of a matrix.
/// @param[in] A Matrix.
/// @param[in] dim 0 for rows, 1 for columns.
/// @return Owned range `[first, last)` and global number.
template <typename T>
Layout index_layout(const MatrixCSR<T>& A, int dim)
{
  std::shared_ptr<const common::IndexMap> map = A.index_map(dim);
  std::array<std::int64_t, 2> r = map->local_range();
  const int bs = A.block_size()[dim];
  return {r[0] * bs, r[1] * bs, map->size_global() * bs};
}

/// @brief Communicator and row and column layouts of a block matrix.
///
/// The layouts are of unrolled indices, in the numbering of the blocks
/// of each block row and block column.
///
/// Every check but the one of owned ranges uses data that is the same
/// on all ranks, so all ranks throw together. Owned ranges can differ
/// on some ranks only; in Debug builds the ranks agree before
/// throwing.
///
/// @param[in] A Blocks of the matrix. The blocks of a block row must
/// have the same rows, and the blocks of a block column the same
/// columns. Every block row and block column must have a block, and all
/// blocks congruent communicators.
/// @return (0) Communicator of the blocks, (1) layout of each block
/// row, and (2) of each block column.
/// @throws std::invalid_argument if `A` does not satisfy the above.
template <typename T>
std::tuple<MPI_Comm, std::vector<Layout>, std::vector<Layout>>
block_layout(Blocks<T> A)
{
  if (A.empty() or A.front().empty())
    throw std::invalid_argument("Block matrix has no blocks.");
  const std::size_t ncols = A.front().size();
  if (std::ranges::any_of(A,
                          [ncols](const std::vector<const MatrixCSR<T>*>& row)
                          { return row.size() != ncols; }))
  {
    throw std::invalid_argument("Block matrix is not rectangular.");
  }

  constexpr Layout unset = {-1, -1, -1};
  std::vector rows(A.size(), unset);
  std::vector cols(ncols, unset);
  MPI_Comm comm = MPI_COMM_NULL;
  bool owned_mismatch = false;
  for (std::size_t i = 0; i < rows.size(); ++i)
  {
    for (std::size_t j = 0; j < ncols; ++j)
    {
      const MatrixCSR<T>* B = A[i][j];
      if (!B)
        continue;

      if (comm == MPI_COMM_NULL)
        comm = B->comm();
      else
      {
        int result = MPI_UNEQUAL;
        int ierr = MPI_Comm_compare(comm, B->comm(), &result);
        dolfinx::MPI::check_error(comm, ierr);
        if (result != MPI_IDENT and result != MPI_CONGRUENT)
        {
          throw std::invalid_argument(
              "Blocks have communicators that are not congruent.");
        }
      }

      Layout r = index_layout(*B, 0);
      Layout c = index_layout(*B, 1);
      if ((rows[i] != unset and rows[i][2] != r[2])
          or (cols[j] != unset and cols[j][2] != c[2]))
      {
        throw std::invalid_argument("Blocks of a block row or block column "
                                    "have different global sizes.");
      }
      owned_mismatch = owned_mismatch or (rows[i] != unset and rows[i] != r)
                       or (cols[j] != unset and cols[j] != c);
      rows[i] = r;
      cols[j] = c;
    }
  }

  if (std::ranges::find(rows, unset) != rows.end()
      or std::ranges::find(cols, unset) != cols.end())
  {
    throw std::invalid_argument(
        "Every block row and block column must have a block.");
  }

#ifndef NDEBUG
  // Agree across ranks, so that no rank is left in a collective
  int mismatch = owned_mismatch;
  int ierr = MPI_Allreduce(MPI_IN_PLACE, &mismatch, 1, MPI_INT, MPI_LOR, comm);
  dolfinx::MPI::check_error(comm, ierr);
  owned_mismatch = mismatch;
#endif
  if (owned_mismatch)
  {
    throw std::invalid_argument(
        "Blocks of a block row or block column have different owned ranges.");
  }

  return {comm, std::move(rows), std::move(cols)};
}

/// @brief Global layout of the rows or columns of a block matrix.
///
/// The owned rows (columns) of the block rows (columns) follow one
/// another on each process, so the first local row (column) is the sum
/// of the first owned rows (columns) of the block rows (columns).
///
/// @param[in] layouts Layout of each block row (column), from
/// block_layout().
/// @return Global index of (0) the first local row (column) and (1) the
/// global number of rows (columns).
std::array<std::int64_t, 2> block_range(std::span<const Layout> layouts)
{
  std::array<std::int64_t, 2> range = {0, 0};
  for (const Layout& l : layouts)
  {
    range[0] += l[0];
    range[1] += l[2];
  }
  return range;
}

/// @brief Number of unrolled entries in the owned rows of a block
/// matrix.
/// @param[in] A Blocks of the matrix, see block_layout().
/// @return Number of entries.
template <typename T>
std::int64_t num_entries(Blocks<T> A)
{
  std::int64_t nnz = 0;
  for (const std::vector<const MatrixCSR<T>*>& row : A)
  {
    for (const MatrixCSR<T>* B : row)
    {
      if (B)
      {
        std::array<int, 2> bs = B->block_size();
        nnz += B->row_ptr()[B->num_owned_rows()] * bs[0] * bs[1];
      }
    }
  }
  return nnz;
}

/// @brief Block column of the only block of a block row.
///
/// The owned entries of such a block row are those of its block, in
/// the same order, so they are unrolled block row by block row.
///
/// @param[in] row Blocks of a block row.
/// @return Block column of the block, or no value if the block row has
/// more than one block.
template <typename T>
std::optional<std::size_t>
single_block(const std::vector<const MatrixCSR<T>*>& row)
{
  std::optional<std::size_t> single;
  for (std::size_t j = 0; j < row.size(); ++j)
  {
    if (!row[j])
      continue;
    if (single)
      return std::nullopt;
    single = j;
  }
  return single;
}

/// @brief Flattened row pointer of the owned rows of a block matrix.
///
/// Each row of a block row has the entries of that row in each of its
/// blocks.
///
/// @param[in] A Blocks of the matrix, see block_layout().
/// @param[in] rows Layout of each block row, from
/// block_layout().
/// @return Row pointer of the unrolled owned rows.
template <typename T>
std::vector<int_t> row_indices(Blocks<T> A, std::span<const Layout> rows)
{
  std::int64_t num_rows_local = 0;
  for (const Layout& r : rows)
    num_rows_local += r[1] - r[0];

  // One block of block size (1, 1): its owned row pointer as is
  if (A.size() == 1 and single_block(A[0]))
  {
    const MatrixCSR<T>* B = A[0][*single_block(A[0])];
    if (B->block_size() == std::array{1, 1})
    {
      const std::vector<std::int64_t>& B_rowptr = B->row_ptr();
      return std::vector<int_t>(
          B_rowptr.begin(), std::next(B_rowptr.begin(), num_rows_local + 1));
    }
  }

  std::vector<int_t> flattened_rowptr(num_rows_local + 1, 0);
  std::int64_t pos = 0;
  for (std::size_t i = 0; i < A.size(); ++i)
  {
    const std::int32_t num_rows = rows[i][1] - rows[i][0];
    if (std::optional<std::size_t> j = single_block(A[i]); j)
    {
      const MatrixCSR<T>* B = A[i][*j];
      std::array<int, 2> bs = B->block_size();
      const std::vector<std::int64_t>& B_rowptr = B->row_ptr();
      for (std::int32_t rb = 0; rb < num_rows / bs[0]; ++rb)
      {
        const int_t delta = (B_rowptr[rb + 1] - B_rowptr[rb]) * bs[1];
        for (int i0 = 0; i0 < bs[0]; ++i0, ++pos)
          flattened_rowptr[pos + 1] = flattened_rowptr[pos] + delta;
      }
      continue;
    }

    for (std::int32_t r = 0; r < num_rows; ++r, ++pos)
    {
      int_t delta = 0;
      for (const MatrixCSR<T>* B : A[i])
      {
        if (!B)
          continue;
        std::array<int, 2> bs = B->block_size();
        const std::vector<std::int64_t>& B_rowptr = B->row_ptr();
        delta += (B_rowptr[r / bs[0] + 1] - B_rowptr[r / bs[0]]) * bs[1];
      }
      flattened_rowptr[pos + 1] = flattened_rowptr[pos] + delta;
    }
  }
  return flattened_rowptr;
}

/// @brief Flattened global column indices of the owned rows of a block
/// matrix, in the layout of block_range().
///
/// The column index map of a block may have more ghosts than the other
/// blocks of its block column, so the ghosts of each block row are
/// numbered with common::stack_index_maps().
///
/// @note Collective if there is more than one block column.
///
/// @param[in] A Blocks of the matrix, see block_layout().
/// @param[in] rows Layout of each block row, from
/// block_layout().
/// @param[in] cols Layout of each block column, from
/// block_layout().
/// @param[in] first_col Global index of the first local column, from
/// block_range().
/// @param[in] nnz Number of local entries, from num_entries().
/// @return Global unrolled column index of each entry, in the order of
/// row_indices().
template <typename T>
std::vector<int_t> col_indices(Blocks<T> A, std::span<const Layout> rows,
                               std::span<const Layout> cols,
                               [[maybe_unused]] std::int64_t first_col,
                               std::int64_t nnz)
{
  const std::size_t ncols = cols.size();

  // Column map and block size of a block of each block column, for the
  // zero blocks of a block row
  std::vector<std::pair<std::reference_wrapper<const common::IndexMap>, int>>
      col_maps;
  for (std::size_t j = 0; j < ncols; ++j)
  {
    for (const std::vector<const MatrixCSR<T>*>& row : A)
    {
      if (const MatrixCSR<T>* B = row[j]; B)
      {
        col_maps.emplace_back(*B->index_map(1), B->block_size()[1]);
        break;
      }
    }
  }

  std::vector<int_t> col_indices(nnz);
  std::int64_t pos = 0;
  for (std::size_t i = 0; i < A.size(); ++i)
  {
    // Global index in the block matrix of the first flattened column of
    // each local column of each block of the block row
    std::vector<std::vector<int_t>> global_cols(ncols);
    if (ncols == 1)
    {
      // The block matrix numbering is that of the block
      const MatrixCSR<T>* B = A[i][0];
      const int bs1 = B->block_size()[1];
      std::vector<std::int64_t> global_indices
          = B->index_map(1)->global_indices();
      global_cols[0].resize(global_indices.size());
      std::ranges::transform(global_indices, global_cols[0].begin(),
                             [bs1](std::int64_t g) -> int_t
                             { return g * bs1; });
    }
    else
    {
      // Ghost columns of the blocks of the block row, in the block
      // matrix numbering. Collective: every rank has the same blocks.
      std::vector<
          std::pair<std::reference_wrapper<const common::IndexMap>, int>>
          maps = col_maps;
      for (std::size_t j = 0; j < ncols; ++j)
      {
        if (const MatrixCSR<T>* B = A[i][j]; B)
          maps[j] = {*B->index_map(1), B->block_size()[1]};
      }
      auto [offset, local_offsets, ghosts, owners]
          = common::stack_index_maps(maps);
      assert(offset == first_col);

      for (std::size_t j = 0; j < ncols; ++j)
      {
        if (!A[i][j])
          continue;
        const common::IndexMap& map = maps[j].first.get();
        const int bs1 = maps[j].second;
        const std::int32_t size_local = map.size_local();
        global_cols[j].resize(size_local + map.num_ghosts());
        for (std::int32_t c = 0; c < size_local; ++c)
          global_cols[j][c] = offset + local_offsets[j] + c * bs1;
        for (std::int32_t g = 0; g < map.num_ghosts(); ++g)
          global_cols[j][size_local + g] = ghosts[j][g * bs1];
      }
    }

    const std::int32_t num_rows = rows[i][1] - rows[i][0];
    if (std::optional<std::size_t> j = single_block(A[i]); j)
    {
      const MatrixCSR<T>* B = A[i][*j];
      std::array<int, 2> bs = B->block_size();
      const std::vector<std::int32_t>& B_cols = B->cols();
      const std::vector<std::int64_t>& B_rowptr = B->row_ptr();
      const std::vector<int_t>& g = global_cols[*j];
      if (bs[0] == 1 and bs[1] == 1)
      {
        // Same layout: map the columns of the block
        std::span<const std::int32_t> owned
            = std::span(B_cols).first(B_rowptr[num_rows]);
        std::ranges::transform(owned, std::next(col_indices.begin(), pos),
                               [&g](std::int32_t c) { return g[c]; });
        pos += owned.size();
        continue;
      }

      for (std::int32_t rb = 0; rb < num_rows / bs[0]; ++rb)
        for (int i0 = 0; i0 < bs[0]; ++i0)
          for (std::int64_t k = B_rowptr[rb]; k < B_rowptr[rb + 1]; ++k)
            for (int i1 = 0; i1 < bs[1]; ++i1)
              col_indices[pos++] = g[B_cols[k]] + i1;
      continue;
    }

    for (std::int32_t r = 0; r < num_rows; ++r)
    {
      for (std::size_t j = 0; j < ncols; ++j)
      {
        const MatrixCSR<T>* B = A[i][j];
        if (!B)
          continue;
        std::array<int, 2> bs = B->block_size();
        const std::vector<std::int32_t>& B_cols = B->cols();
        const std::vector<std::int64_t>& B_rowptr = B->row_ptr();
        for (std::int64_t k = B_rowptr[r / bs[0]]; k < B_rowptr[r / bs[0] + 1];
             ++k)
        {
          for (int i1 = 0; i1 < bs[1]; ++i1)
            col_indices[pos++] = global_cols[j][B_cols[k]] + i1;
        }
      }
    }
  }
  assert(pos == nnz);
  return col_indices;
}

/// @brief Flattened values of the owned rows of a block matrix.
/// @param[in] A Blocks of the matrix, see block_layout().
/// @param[in] rows Layout of each block row, from
/// block_layout().
/// @param[in] nnz Number of local entries, from num_entries().
/// @return Value of each entry, in the order of row_indices().
template <typename T>
std::vector<T> matrix_values(Blocks<T> A, std::span<const Layout> rows,
                             std::int64_t nnz)
{
  // One block of block size (1, 1): its owned values as is
  if (A.size() == 1 and single_block(A[0]))
  {
    const MatrixCSR<T>* B = A[0][*single_block(A[0])];
    if (B->block_size() == std::array{1, 1})
    {
      const std::vector<T>& B_values = B->values();
      return std::vector<T>(B_values.begin(), std::next(B_values.begin(), nnz));
    }
  }

  std::vector<T> flattened_values(nnz);
  std::int64_t pos = 0;
  for (std::size_t i = 0; i < A.size(); ++i)
  {
    const std::int32_t num_rows = rows[i][1] - rows[i][0];
    if (std::optional<std::size_t> j = single_block(A[i]); j)
    {
      const MatrixCSR<T>* B = A[i][*j];
      std::array<int, 2> bs = B->block_size();
      const std::vector<T>& B_values = B->values();
      const std::vector<std::int64_t>& B_rowptr = B->row_ptr();
      if (bs[0] == 1 and bs[1] == 1)
      {
        // Same layout: copy as a whole
        std::copy_n(B_values.begin(), B_rowptr[num_rows],
                    std::next(flattened_values.begin(), pos));
        pos += B_rowptr[num_rows];
        continue;
      }

      for (std::int32_t rb = 0; rb < num_rows / bs[0]; ++rb)
      {
        for (int i0 = 0; i0 < bs[0]; ++i0)
        {
          for (std::int64_t k = B_rowptr[rb]; k < B_rowptr[rb + 1]; ++k)
          {
            for (int i1 = 0; i1 < bs[1]; ++i1)
            {
              flattened_values[pos++]
                  = B_values[k * bs[0] * bs[1] + i0 * bs[1] + i1];
            }
          }
        }
      }
      continue;
    }

    for (std::int32_t r = 0; r < num_rows; ++r)
    {
      for (const MatrixCSR<T>* B : A[i])
      {
        if (!B)
          continue;
        std::array<int, 2> bs = B->block_size();
        const std::vector<T>& B_values = B->values();
        const std::vector<std::int64_t>& B_rowptr = B->row_ptr();
        const int i0 = r % bs[0];
        for (std::int64_t k = B_rowptr[r / bs[0]]; k < B_rowptr[r / bs[0] + 1];
             ++k)
        {
          for (int i1 = 0; i1 < bs[1]; ++i1)
          {
            flattened_values[pos++]
                = B_values[k * bs[0] * bs[1] + i0 * bs[1] + i1];
          }
        }
      }
    }
  }
  assert(pos == nnz);
  return flattened_values;
}
} // namespace
//----------------------------------------------------------------------------
template <typename T>
SuperLUDistMatrix<T>::SuperLUDistMatrix(const MatrixCSR<T>& A)
    : SuperLUDistMatrix(std::vector<std::vector<const MatrixCSR<T>*>>{{&A}})
{
}
//----------------------------------------------------------------------------
template <typename T>
SuperLUDistMatrix<T>::SuperLUDistMatrix(
    std::span<const std::vector<const MatrixCSR<T>*>> A)
    : _comm(MPI_COMM_NULL, false)
{
  // Validate locally, before the first collective (the duplication of
  // the communicator)
  auto [comm, rows, cols] = block_layout(A);
  const std::int64_t nnz = num_entries(A);
  check_int_t(nnz);

  auto [first_row, m] = block_range(rows);
  auto [first_col, n] = block_range(cols);
  _comm = dolfinx::MPI::Comm(comm);

  _rowptr
      = std::make_unique<SuperLUDistStructs::vec_int_t>(row_indices(A, rows));
  _cols = std::make_unique<SuperLUDistStructs::vec_int_t>(
      col_indices(A, rows, cols, first_col, nnz));
  _matA_values = matrix_values(A, rows, nnz);
  _supermatrix = create_supermatrix<T>(m, n, _rowptr->vec.size() - 1, first_row,
                                       _matA_values, *_rowptr, *_cols);
}
//----------------------------------------------------------------------------
/// @brief Get MPI communicator that matrix is defined on.
template <typename T>
MPI_Comm SuperLUDistMatrix<T>::comm() const
{
  return _comm.comm();
}
//----------------------------------------------------------------------------
template <typename T>
SuperLUDistStructs::SuperMatrix* SuperLUDistMatrix<T>::supermatrix() const
{
  return _supermatrix.get();
}
//----------------------------------------------------------------------------
template class la::SuperLUDistMatrix<double>;
template class la::SuperLUDistMatrix<float>;
template class la::SuperLUDistMatrix<std::complex<double>>;
//----------------------------------------------------------------------------

namespace
{
template <typename V, typename W>
void option_setter(W& option, std::string_view value_in,
                   std::string_view option_name,
                   std::initializer_list<V> values,
                   std::initializer_list<std::string_view> value_names)
{
  // TODO: Can be done nicely with std::views::zip in C++23.
  for (auto i : std::views::iota(std::size_t{0}, value_names.size()))
  {
    if (value_in == *(value_names.begin() + i))
    {
      option = *(values.begin() + i);
      logger().info("Set {} to {}", option_name, value_in);
      return;
    }
  }
  throw std::invalid_argument("Unsupported value");
}
} // namespace

//----------------------------------------------------------------------------
// Trick for declaring anonymous typedef structs from SuperLU_DIST
struct dolfinx::la::SuperLUDistStructs::gridinfo_t : public ::gridinfo_t
{
};
//----------------------------------------------------------------------------
struct dolfinx::la::SuperLUDistStructs::superlu_dist_options_t
    : public ::superlu_dist_options_t
{
};
//----------------------------------------------------------------------------
struct dolfinx::la::SuperLUDistStructs::dScalePermstruct_t
    : public ::dScalePermstruct_t
{
};
//----------------------------------------------------------------------------
struct dolfinx::la::SuperLUDistStructs::sScalePermstruct_t
    : public ::sScalePermstruct_t
{
};
//----------------------------------------------------------------------------
struct dolfinx::la::SuperLUDistStructs::zScalePermstruct_t
    : public ::zScalePermstruct_t
{
};
//----------------------------------------------------------------------------
struct dolfinx::la::SuperLUDistStructs::dLUstruct_t : public ::dLUstruct_t
{
};
//----------------------------------------------------------------------------
struct dolfinx::la::SuperLUDistStructs::sLUstruct_t : public ::sLUstruct_t
{
};
//----------------------------------------------------------------------------
struct dolfinx::la::SuperLUDistStructs::zLUstruct_t : public ::zLUstruct_t
{
};
//----------------------------------------------------------------------------
struct dolfinx::la::SuperLUDistStructs::dSOLVEstruct_t : public ::dSOLVEstruct_t
{
};
//----------------------------------------------------------------------------
struct dolfinx::la::SuperLUDistStructs::sSOLVEstruct_t : public ::sSOLVEstruct_t
{
};
//----------------------------------------------------------------------------
struct dolfinx::la::SuperLUDistStructs::zSOLVEstruct_t : public ::zSOLVEstruct_t
{
};
//----------------------------------------------------------------------------
void GridInfoDeleter::operator()(
    SuperLUDistStructs::gridinfo_t* g) const noexcept
{
  superlu_gridexit(g);
  delete g;
};
//----------------------------------------------------------------------------
void ScalePermStructDeleter::operator()(
    SuperLUDistStructs::dScalePermstruct_t* s) const noexcept
{
  dScalePermstructFree(s);
  delete s;
};
//----------------------------------------------------------------------------
void ScalePermStructDeleter::operator()(
    SuperLUDistStructs::sScalePermstruct_t* s) const noexcept
{
  sScalePermstructFree(s);
  delete s;
};
//----------------------------------------------------------------------------
void ScalePermStructDeleter::operator()(
    SuperLUDistStructs::zScalePermstruct_t* s) const noexcept
{
  zScalePermstructFree(s);
  delete s;
};
//----------------------------------------------------------------------------
void LUStructDeleter::operator()(
    SuperLUDistStructs::dLUstruct_t* l) const noexcept
{
  dLUstructFree(l);
  delete l;
};
//----------------------------------------------------------------------------
void LUStructDeleter::operator()(
    SuperLUDistStructs::sLUstruct_t* l) const noexcept
{
  sLUstructFree(l);
  delete l;
};
//----------------------------------------------------------------------------
void LUStructDeleter::operator()(
    SuperLUDistStructs::zLUstruct_t* l) const noexcept
{
  zLUstructFree(l);
  delete l;
};
//----------------------------------------------------------------------------
void SolveStructDeleter::operator()(
    SuperLUDistStructs::dSOLVEstruct_t* s) const noexcept
{
  if (o->SolveInitialized)
    dSolveFinalize(o, s);
  delete s;
};
//----------------------------------------------------------------------------
void SolveStructDeleter::operator()(
    SuperLUDistStructs::sSOLVEstruct_t* s) const noexcept
{
  if (o->SolveInitialized)
    sSolveFinalize(o, s);
  delete s;
};
//----------------------------------------------------------------------------
void SolveStructDeleter::operator()(
    SuperLUDistStructs::zSOLVEstruct_t* s) const noexcept
{
  if (o->SolveInitialized)
    zSolveFinalize(o, s);
  delete s;
};
//----------------------------------------------------------------------------
template <typename T>
SuperLUDistSolver<T>::SuperLUDistSolver(
    std::shared_ptr<const SuperLUDistMatrix<T>> A)
    : _superlu_matA(std::move(A)),
      _options(
          []
          {
            auto o = std::make_unique<
                SuperLUDistStructs::superlu_dist_options_t>();
            set_default_options_dist(o.get());
            o->PrintStat = NO;
#ifdef HAVE_PARMETIS
            // METIS_AT_PLUS_A deadlocks if ranks get different orderings.
            // ParSymbFact stays off as it can hang, see
            // https://github.com/xiaoyeli/superlu_dist/pull/231
            o->ColPerm = PARMETIS;
#endif
            return o;
          }()),
      _gridinfo(
          [comm = _superlu_matA->comm()]
          {
            // Computes a balanced 2D process grid - see PETSc interface
            int size = dolfinx::MPI::size(comm);
            int nprow = static_cast<int>(std::sqrt(static_cast<double>(size)));
            while (size % nprow != 0)
              --nprow;
            int npcol = size / nprow;
            std::unique_ptr<SuperLUDistStructs::gridinfo_t, GridInfoDeleter> p(
                new SuperLUDistStructs::gridinfo_t, GridInfoDeleter{});
            superlu_gridinit(comm, nprow, npcol, p.get());
            return p;
          }()),
      _scalepermstruct(
          [m = _superlu_matA->supermatrix()->nrow]
          {
            std::unique_ptr<typename map_t<T>::ScalePermstruct_t,
                            ScalePermStructDeleter>
                s(new map_t<T>::ScalePermstruct_t, ScalePermStructDeleter{});

            if constexpr (std::is_same_v<T, double>)
            {
              dScalePermstructInit(m, m, s.get());
            }
            else if constexpr (std::is_same_v<T, float>)
            {
              sScalePermstructInit(m, m, s.get());
            }
            else if constexpr (std::is_same_v<T, std::complex<double>>)
            {
              zScalePermstructInit(m, m, s.get());
            }
            else
              static_assert(impl::always_false_v<T>, "Invalid scalar type");
            return s;
          }()),
      _lustruct(
          [m = _superlu_matA->supermatrix()->nrow]
          {
            std::unique_ptr<typename map_t<T>::LUstruct_t, LUStructDeleter> l(
                new map_t<T>::LUstruct_t, LUStructDeleter{});

            if constexpr (std::is_same_v<T, double>)
            {
              dLUstructInit(m, l.get());
            }
            else if constexpr (std::is_same_v<T, float>)
            {
              sLUstructInit(m, l.get());
            }
            else if constexpr (std::is_same_v<T, std::complex<double>>)
            {
              zLUstructInit(m, l.get());
            }
            else
              static_assert(impl::always_false_v<T>, "Invalid scalar type");
            return l;
          }()),
      _solvestruct(new typename map_t<T>::SOLVEstruct_t{},
                   SolveStructDeleter{_options.get()})
{
}
//----------------------------------------------------------------------------
template <typename T>
void SuperLUDistSolver<T>::set_options(
    const SuperLUDistStructs::superlu_dist_options_t& options)
{
  *_options = options;
}
//----------------------------------------------------------------------------
template <typename T>
void SuperLUDistSolver<T>::set_option(std::string_view name,
                                      std::string_view value)
{
  logger().info("Attempting to set option {} to {}", name, value);

  // Boolean-valued options, keyed to their member in superlu_dist_options_t.
  // Internal state flags managed by the solver (e.g. SolveInitialized,
  // RefineInitialized) are deliberately excluded: setting them by hand can
  // trigger cleanup of uninitialised SuperLU_DIST structures.
  using bool_option_t = decltype(&::superlu_dist_options_t::Equil);
  static constexpr std::array<std::pair<std::string_view, bool_option_t>, 12>
      bool_options{
          {{"Equil", &::superlu_dist_options_t::Equil},
           {"ParSymbFact", &::superlu_dist_options_t::ParSymbFact},
           {"DiagInv", &::superlu_dist_options_t::DiagInv},
           {"SymmetricMode", &::superlu_dist_options_t::SymmetricMode},
           {"PivotGrowth", &::superlu_dist_options_t::PivotGrowth},
           {"ConditionNumber", &::superlu_dist_options_t::ConditionNumber},
           {"ReplaceTinyPivot", &::superlu_dist_options_t::ReplaceTinyPivot},
           {"PrintStat", &::superlu_dist_options_t::PrintStat},
           {"lookahead_etree", &::superlu_dist_options_t::lookahead_etree},
           {"SymPattern", &::superlu_dist_options_t::SymPattern},
           {"Use_TensorCore", &::superlu_dist_options_t::Use_TensorCore},
           {"Algo3d", &::superlu_dist_options_t::Algo3d}}};

  // Search the boolean options first
  if (auto it = std::ranges::find_if(bool_options, [name](const auto& e)
                                     { return e.first == name; });
      it != bool_options.end())
  {
    yes_no_t& field = _options.get()->*(it->second);
    if (value == "YES")
      field = YES;
    else if (value == "NO")
      field = NO;
    else
    {
      throw std::invalid_argument(
          "'Boolean' option values must be string 'YES' or 'NO'");
    }
    logger().info("Set {} to {}", name, value);
    return;
  }

  // Search some enum types
  if (name == "Fact")
  {
    option_setter(
        _options->Fact, value, "Fact",
        {DOFACT, SamePattern, SamePattern_SameRowPerm, FACTORED},
        {"DOFACT", "SamePattern", "SamePattern_SameRowPerm", "FACTORED"});
  }
  else if (name == "Trans")
  {
    option_setter(_options->Trans, value, "Trans", {NOTRANS, TRANS, CONJ},
                  {"NOTRANS", "TRANS", "CONJ"});
  }
  else if (name == "ColPerm")
  {
    option_setter(_options->ColPerm, value, "ColPerm",
                  {NATURAL, MMD_ATA, MMD_AT_PLUS_A, COLAMD, METIS_AT_PLUS_A,
                   PARMETIS, METIS_ATA, ZOLTAN, MY_PERMC},
                  {"NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD",
                   "METIS_AT_PLUS_A", "PARMETIS", "METIS_ATA", "ZOLTAN",
                   "MY_PERMC"});
    if (_options->ColPerm != PARMETIS and _options->ParSymbFact == YES)
    {
      _options->ParSymbFact = NO;
      logger().info("Set ParSymbFact to NO as it requires ColPerm PARMETIS");
    }
  }
  else if (name == "RowPerm")
  {
    option_setter(
        _options->RowPerm, value, "RowPerm",
        {NOROWPERM, LargeDiag_MC64, LargeDiag_HWPM, MY_PERMR},
        {"NOROWPERM", "LargeDiag_MC64", "LargeDiag_HWPM", "MY_PERMR"});
  }
  else
  {
    throw std::invalid_argument("Unsupported option name");
  }
}
//----------------------------------------------------------------------------
template <typename T>
SuperLUDistSolver<T>::~SuperLUDistSolver()
{
  if (_factored)
  {
    int_t n = _superlu_matA->supermatrix()->ncol;
    if constexpr (std::is_same_v<T, double>)
      dDestroy_LU(n, _gridinfo.get(), _lustruct.get());
    else if constexpr (std::is_same_v<T, float>)
      sDestroy_LU(n, _gridinfo.get(), _lustruct.get());
    else if constexpr (std::is_same_v<T, std::complex<double>>)
      zDestroy_LU(n, _gridinfo.get(), _lustruct.get());
    else
      static_assert(impl::always_false_v<T>, "Invalid scalar type");
  }
}
//----------------------------------------------------------------------------
template <typename T>
void SuperLUDistSolver<T>::set_A(std::shared_ptr<const SuperLUDistMatrix<T>> A,
                                 std::string_view fact)
{
  if (A->supermatrix()->nrow != _superlu_matA->supermatrix()->nrow
      or A->supermatrix()->ncol != _superlu_matA->supermatrix()->ncol)
  {
    throw std::invalid_argument(
        "New matrix A has different size to the matrix used to construct the "
        "solver.");
  }

  // The process grid is built from the original matrix's communicator and is
  // reused, so the new matrix must live on a compatible communicator.
  int comm_cmp;
  MPI_Comm_compare(A->comm(), _superlu_matA->comm(), &comm_cmp);
  if (comm_cmp != MPI_IDENT and comm_cmp != MPI_CONGRUENT)
  {
    throw std::invalid_argument(
        "New matrix A is defined on a different MPI communicator to the "
        "matrix used to construct the solver.");
  }
  _superlu_matA = A;

  // See pddistribute in SuperLU_DIST: on Fact=DOFACT or Fact=SamePattern
  // pdgssvx overwrites LUstruct's internal pointers without freeing the
  // previous arrays, so a prior factorisation must be released first.
  // Fact=SamePattern_SameRowPerm reuses the existing arrays in place.
  if (fact == "DOFACT" or fact == "SamePattern")
  {
    if (_factored)
    {
      int_t n = _superlu_matA->supermatrix()->ncol;
      if constexpr (std::is_same_v<T, double>)
        dDestroy_LU(n, _gridinfo.get(), _lustruct.get());
      else if constexpr (std::is_same_v<T, float>)
        sDestroy_LU(n, _gridinfo.get(), _lustruct.get());
      else if constexpr (std::is_same_v<T, std::complex<double>>)
        zDestroy_LU(n, _gridinfo.get(), _lustruct.get());
      else
        static_assert(impl::always_false_v<T>, "Invalid scalar type");
      _factored = false;
    }

    // SamePattern crashes SuperLU_DIST with ParSymbFact
    bool dofact = fact == "DOFACT" or _options->ParSymbFact == YES;
    if (dofact)
    {
      if constexpr (std::is_same_v<T, double>)
        dSolveFinalize(_options.get(), _solvestruct.get());
      else if constexpr (std::is_same_v<T, float>)
        sSolveFinalize(_options.get(), _solvestruct.get());
      else if constexpr (std::is_same_v<T, std::complex<double>>)
        zSolveFinalize(_options.get(), _solvestruct.get());
      else
        static_assert(impl::always_false_v<T>, "Invalid scalar type");
    }
    _options->Fact = dofact ? DOFACT : SamePattern;
  }
  else if (fact == "SamePattern_SameRowPerm")
  {
    _options->Fact = SamePattern_SameRowPerm;
  }
  else
  {
    throw std::invalid_argument("set_A fact must be one of 'DOFACT', "
                                "'SamePattern', 'SamePattern_SameRowPerm'");
  }
}
//----------------------------------------------------------------------------
template <typename T>
int SuperLUDistSolver<T>::solve(const la::Vector<T>& b, la::Vector<T>& u)
{
  common::Timer tsolve("SuperLU_DIST solve");

  if (b.array().size() != u.array().size())
    throw std::invalid_argument("b and u have incompatible size/layout");
  const int_t m_loc_A
      = static_cast<NRformat_loc*>(_superlu_matA->supermatrix()->Store)->m_loc;
  if (b.index_map()->size_local() * b.bs() != m_loc_A
      or u.index_map()->size_local() * u.bs() != m_loc_A)
  {
    throw std::invalid_argument(std::format(
        "Owned sizes of b ({}) and u ({}) differ from the {} owned rows of "
        "A.",
        b.index_map()->size_local() * b.bs(),
        u.index_map()->size_local() * u.bs(), m_loc_A));
  }

  if (_options->ParSymbFact == YES and _options->ColPerm != PARMETIS)
    throw std::invalid_argument("ParSymbFact YES requires ColPerm PARMETIS");
  if (_options->ParSymbFact == YES and _options->Fact == SamePattern)
  {
    throw std::invalid_argument(
        "ParSymbFact YES does not support Fact SamePattern");
  }

  // Warn if Factor is DOFACT but solve already appears to be initialised.
  if (_options->Fact == DOFACT && _options->SolveInitialized == YES)
  {
    logger().warn("Extra call to solve with option Fact set to DOFACT. "
                  "This leads to incorrect results; try FACTORED.");
  }

  int_t m_loc
      = static_cast<NRformat_loc*>(_superlu_matA->supermatrix()->Store)->m_loc;
  // RHS
  int_t ldb = m_loc;
  // TODO: Support for multiple right-hand sides?
  int_t nrhs = 1;

  int info = 0;
  SuperLUStat_t stat;
  PStatInit(&stat);

  // Copy b to u (SuperLU_DIST reads b from u and then overwrites u with
  // solution)
  std::copy_n(b.array().begin(), b.array().size(), u.array().begin());

  std::vector<scalar_value_t<T>> berr(nrhs);
  if constexpr (std::is_same_v<T, double>)
  {
    logger().info("Start solve [float64]");

    logger().info("Call SuperLU_DIST pdgssvx()");
    pdgssvx(_options.get(), _superlu_matA->supermatrix(),
            _scalepermstruct.get(), u.array().data(), ldb, nrhs,
            _gridinfo.get(), _lustruct.get(), _solvestruct.get(), berr.data(),
            &stat, &info);
  }
  else if constexpr (std::is_same_v<T, float>)
  {
    logger().info("Start solve [float32]");

    logger().info("Call SuperLU_DIST psgssvx()");
    psgssvx(_options.get(), _superlu_matA->supermatrix(),
            _scalepermstruct.get(), u.array().data(), ldb, nrhs,
            _gridinfo.get(), _lustruct.get(), _solvestruct.get(), berr.data(),
            &stat, &info);
  }
  else if constexpr (std::is_same_v<T, std::complex<double>>)
  {
    logger().info("Start solve [complex128]");

    logger().info("Call SuperLU_DIST pzgssvx()");
    pzgssvx(_options.get(), _superlu_matA->supermatrix(),
            _scalepermstruct.get(),
            reinterpret_cast<doublecomplex*>(u.array().data()), ldb, nrhs,
            _gridinfo.get(), _lustruct.get(), _solvestruct.get(), berr.data(),
            &stat, &info);
  }
  else
    static_assert(impl::always_false_v<T>, "Invalid scalar type");

  if (info != 0)
    logger().warn("SuperLU_DIST p*gssvx() returned info = {}", info);

  // Record the factorisation so Destroy_LU is called on the LUstruct
  // internals. info < 0 means an illegal argument, where pdgssvx returns
  // before allocating anything.
  if (info >= 0)
    _factored = true;

  PStatPrint(_options.get(), &stat, _gridinfo.get());
  PStatFree(&stat);

  logger().info("Finished solve");

  return info;
}
//----------------------------------------------------------------------------
template class la::SuperLUDistSolver<double>;
template class la::SuperLUDistSolver<float>;
template class la::SuperLUDistSolver<std::complex<double>>;
//----------------------------------------------------------------------------
#endif
