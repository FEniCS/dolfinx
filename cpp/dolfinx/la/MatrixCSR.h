// Copyright (C) 2021-2022 Garth N. Wells and Chris N. Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "SparsityPattern.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace dolfinx::la
{
namespace impl
{
/// @brief Set data in a CSR matrix.
///
/// @param[out] data The CSR matrix data
/// @param[in] cols The CSR column indices
/// @param[in] row_ptr The pointer to the ith row in the CSR data
/// @param[in] x The `m` by `n` dense block of values (row-major) to set
/// in the matrix
/// @param[in] xrows The row indices of `x`
/// @param[in] xcols The column indices of `x`
/// @param[in] local_size The maximum row index that can be set. Used
/// when debugging is own to check that rows beyond a permitted range
/// are not being set.
template <int BS0, int BS1, typename U, typename V, typename W, typename X,
          typename Y>
void set_csr(U&& data, const V& cols, const W& row_ptr, const X& x,
             const Y& xrows, const Y& xcols, typename Y::value_type local_size);

/// Set blocked data with given block sizes into a non-blocked MatrixCSR (bs=1)
/// Matrix sparsity must be correct to accept the data
template <int BS0, int BS1, typename U, typename V, typename W, typename X,
          typename Y>
void set_blocked_csr(U&& data, const V& cols, const W& row_ptr, const X& x,
                     const Y& xrows, const Y& xcols,
                     typename Y::value_type local_size);

/// @brief Add data to a CSR matrix
///
/// @tparam BS0 Row block size (of both matrix and data)
/// @tparam BS1 Column block size (of both matrix and data)
/// @param[out] data The CSR matrix data
/// @param[in] cols The CSR column indices
/// @param[in] row_ptr The pointer to the ith row in the CSR data
/// @param[in] x The `m` by `n` dense block of values (row-major) to add
/// to the matrix
/// @param[in] xrows The row indices of `x`
/// @param[in] xcols The column indices of `x`
///
/// @note In the case of block data, where BS0 or BS1 are greater than
/// one, the layout of the input data is still the same. For example, the
/// following can be inserted into the top-left corner
/// with xrows={0,1} and xcols={0,1}, BS0=2, BS1=2 and
/// x={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}.
///
/// 0  1  | 2  3
/// 4  5  | 6  7
/// -------------
/// 8  9  | 10 11
/// 12 13 | 14 15
template <int BS0, int BS1, typename U, typename V, typename W, typename X,
          typename Y>
void add_csr(U&& data, const V& cols, const W& row_ptr, const X& x,
             const Y& xrows, const Y& xcols);

/// Add blocked data into a non-blocked matrix (Matrix block size = 1)
/// @note see `add_csr` for data layout
///
/// @tparam BS0 Row block size of Data
/// @tparam BS1 Column block size of Data
template <int BS0, int BS1, typename U, typename V, typename W, typename X,
          typename Y>
void add_blocked_csr(U&& data, const V& cols, const W& row_ptr, const X& x,
                     const Y& xrows, const Y& xcols);

/// Add non-blocked data into a blocked matrix (Data block size = 1)
/// @param bs0 Row block size of Matrix
/// @param bs1 Column block size of Matrix
template <typename U, typename V, typename W, typename X, typename Y>
void add_nonblocked_csr(U&& data, const V& cols, const W& row_ptr, const X& x,
                        const Y& xrows, const Y& xcols, int bs0, int bs1);
} // namespace impl

/// @brief Modes for representing block structured matrices
enum class BlockMode : int
{
  compact = 0, /// Each entry in the sparsity pattern of the matrix refers to a
               /// block of data of size (bs[0], bs[1])
  expanded = 1 /// The sparsity pattern is expanded by (bs[0], bs[1]), and each
               /// entry refers to one data item.
};

/// Distributed sparse matrix
///
/// The matrix storage format is compressed sparse row. The matrix is
/// partitioned row-wise across MPI rank.
///
/// @warning Highly experimental storage of a matrix in CSR format which
/// can be assembled into using the usual DOLFINx assembly routines
/// Matrix internal data can be accessed for interfacing with other
/// code.
/// @todo Handle block sizes
///
/// @tparam Scalar Scalar type of matrix entries
/// @tparam Container Sequence container type to store matrix entries
/// @tparam ColContainer Column index container type
/// @tparam RowPtrContainer Row pointer container type
template <class Scalar, class Container = std::vector<Scalar>,
          class ColContainer = std::vector<std::int32_t>,
          class RowPtrContainer = std::vector<std::int64_t>>
class MatrixCSR
{
  static_assert(std::is_same_v<typename Container::value_type, Scalar>);

public:
  /// Scalar type
  using value_type = Scalar;

  /// Matrix entries container type
  using container_type = Container;

  /// Column index container type
  using column_container_type = ColContainer;

  /// Row pointer container type
  using rowptr_container_type = RowPtrContainer;

  static_assert(std::is_same_v<value_type, typename container_type::value_type>,
                "Scalar type and container value type must be the same.");

  /// @brief Insertion functor for setting values in matrix. It is
  /// typically used in finite element assembly functions.
  ///
  /// @tparam BS0 Row block size for insertion
  /// @tparam BS1 Column block size for insertion
  ///
  /// @note Block size of matrix may be (1, 1) or (BS0, BS1)
  ///
  /// @return Function for inserting values into `A`
  /// @todo clarify setting on non-owned entries
  template <int BS0 = 1, int BS1 = 1>
  auto mat_set_values()
  {
    if ((BS0 != _bs[0] and BS0 > 1 and _bs[0] > 1)
        or (BS1 != _bs[1] and BS1 > 1 and _bs[1] > 1))
      throw std::runtime_error(
          "Cannot insert blocks of different size than matrix block size");

    return [&](std::span<const std::int32_t> rows,
               std::span<const std::int32_t> cols,
               std::span<const value_type> data) -> int
    {
      this->set<BS0, BS1>(data, rows, cols);
      return 0;
    };
  }

  /// @brief Insertion functor for accumulating values in matrix. It is
  /// typically used in finite element assembly functions.
  ///
  /// @tparam BS0 Row block size for insertion
  /// @tparam BS1 Column block size for insertion
  ///
  /// @note Block size of matrix may be (1, 1) or (BS0, BS1)
  ///
  /// @return Function for inserting values into `A`
  template <int BS0 = 1, int BS1 = 1>
  auto mat_add_values()
  {
    if ((BS0 != _bs[0] and BS0 > 1 and _bs[0] > 1)
        or (BS1 != _bs[1] and BS1 > 1 and _bs[1] > 1))
      throw std::runtime_error(
          "Cannot insert blocks of different size than matrix block size");

    return [&](std::span<const std::int32_t> rows,
               std::span<const std::int32_t> cols,
               std::span<const value_type> data) -> int
    {
      this->add<BS0, BS1>(data, rows, cols);
      return 0;
    };
  }

  /// @brief Create a distributed matrix.
  /// @param[in] p The sparsity pattern the describes the parallel
  /// @param[in] mode Block mode, when block size > 1.
  /// distribution and the non-zero structure.
  MatrixCSR(const SparsityPattern& p, BlockMode mode = BlockMode::compact);

  /// Move constructor
  /// @todo Check handling of MPI_Request
  MatrixCSR(MatrixCSR&& A) = default;

  /// @brief Set all non-zero local entries to a value including entries
  /// in ghost rows.
  /// @param[in] x The value to set non-zero matrix entries to
  void set(value_type x) { std::fill(_data.begin(), _data.end(), x); }

  /// @brief Set values in the matrix.
  /// @note Only entries included in the sparsity pattern used to
  /// initialize the matrix can be set
  /// @note All indices are local to the calling MPI rank and entries
  /// cannot be set in ghost rows.
  /// @note This should be called after `finalize`. Using before
  /// `finalize` will set the values correctly, but incoming values may
  /// get added to them during a subsequent finalize operation.
  /// @param[in] x The `m` by `n` dense block of values (row-major) to
  /// set in the matrix
  /// @param[in] rows The row indices of `x`
  /// @param[in] cols The column indices of `x`
  template <int BS0, int BS1>
  void set(std::span<const value_type> x, std::span<const std::int32_t> rows,
           std::span<const std::int32_t> cols)
  {
    if (_bs[0] == BS0 and _bs[1] == BS1)
      impl::set_csr<BS0, BS1>(_data, _cols, _row_ptr, x, rows, cols,
                              _index_maps[0]->size_local());
    else if (_bs[0] == 1 and _bs[1] == 1)
      impl::set_blocked_csr<BS0, BS1>(_data, _cols, _row_ptr, x, rows, cols,
                                      _index_maps[0]->size_local());
    else
      throw std::runtime_error(
          "Insertion with BS=1 into MatrixCSR with bs>1 not yet implemented");
  }

  /// @brief Accumulate values in the matrix
  /// @note Only entries included in the sparsity pattern used to
  /// initialize the matrix can be accumulated in to
  /// @note All indices are local to the calling MPI rank and entries
  /// may go into ghost rows.
  /// @note Use `finalize` after all entries have been added to send
  /// ghost rows to owners. Adding more entries after `finalize` is
  /// allowed, but another call to `finalize` will then be required.
  ///
  /// @tparam BS0 Row block size of data
  /// @tparam BS1 Column block size of data
  /// @param[in] x The `m` by `n` dense block of values (row-major) to
  /// add to the matrix
  /// @param[in] rows The row indices of `x`
  /// @param[in] cols The column indices of `x`
  template <int BS0 = 1, int BS1 = 1>
  void add(std::span<const value_type> x, std::span<const std::int32_t> rows,
           std::span<const std::int32_t> cols)
  {
    assert(x.size() == rows.size() * cols.size() * BS0 * BS1);
    if (_bs[0] == BS0 and _bs[1] == BS1)
    {
      impl::add_csr<BS0, BS1>(_data, _cols, _row_ptr, x, rows, cols);
    }
    else if (_bs[0] == 1 and _bs[1] == 1)
    {
      // Add blocked data to a regular CSR matrix (_bs[0]=1, _bs[1]=1)
      impl::add_blocked_csr<BS0, BS1>(_data, _cols, _row_ptr, x, rows, cols);
    }
    else
    {
      assert(BS0 == 1 and BS1 == 1);
      // Add non-blocked data to a blocked CSR matrix (BS0=1, BS1=1)
      impl::add_nonblocked_csr(_data, _cols, _row_ptr, x, rows, cols, _bs[0],
                               _bs[1]);
    }
  }

  /// Number of local rows excluding ghost rows
  std::int32_t num_owned_rows() const { return _index_maps[0]->size_local(); }

  /// Number of local rows including ghost rows
  std::int32_t num_all_rows() const { return _row_ptr.size() - 1; }

  /// Copy to a dense matrix
  /// @note This function is typically used for debugging and not used
  /// in production
  /// @note Ghost rows are also returned, and these can be truncated
  /// manually by using num_owned_rows() if required.
  /// @return Dense copy of the part of the matrix on the calling rank.
  /// Storage is row-major.
  std::vector<value_type> to_dense() const;

  /// @brief Transfer ghost row data to the owning ranks accumulating
  /// received values on the owned rows, and zeroing any existing data
  /// in ghost rows.
  void finalize()
  {
    finalize_begin();
    finalize_end();
  }

  /// @brief Begin transfer of ghost row data to owning ranks, where it
  /// will be accumulated into existing owned rows.
  /// @note Calls to this function must be followed by
  /// MatrixCSR::finalize_end(). Between the two calls matrix values
  /// must not be changed.
  /// @note This function does not change the matrix data. Data update only
  /// occurs with `finalize_end()`.
  void finalize_begin();

  /// @brief End transfer of ghost row data to owning ranks.
  /// @note Must be preceded by MatrixCSR::finalize_begin()
  /// @note Matrix data received from other processes will be
  /// accumulated into locally owned rows, and ghost rows will be
  /// zeroed.
  void finalize_end();

  /// Compute the Frobenius norm squared
  double norm_squared() const;

  /// @brief Index maps for the row and column space.
  ///
  /// The row IndexMap contains ghost entries for rows which may be
  /// inserted into and the column IndexMap contains all local and ghost
  /// columns that may exist in the owned rows.
  ///
  /// @return Row (0) and column (1) index maps
  const std::array<std::shared_ptr<const common::IndexMap>, 2>&
  index_maps() const
  {
    return _index_maps;
  }

  /// Get local data values
  /// @note Includes ghost values
  container_type& values() { return _data; }

  /// Get local values (const version)
  /// @note Includes ghost values
  const container_type& values() const { return _data; }

  /// Get local row pointers
  /// @note Includes pointers to ghost rows
  const rowptr_container_type& row_ptr() const { return _row_ptr; }

  /// Get local column indices
  /// @note Includes columns in ghost rows
  const column_container_type& cols() const { return _cols; }

  /// Get the start of off-diagonal (unowned columns) on each row,
  /// allowing the matrix to be split (virtually) into two parts.
  /// Operations (such as matrix-vector multiply) between the owned
  /// parts of the matrix and vector can then be performed separately
  /// from operations on the unowned parts.
  /// @note Includes ghost rows, which should be truncated manually if
  /// not required.
  const rowptr_container_type& off_diag_offset() const
  {
    return _off_diagonal_offset;
  }

  /// Block size
  /// @return block sizes for rows and columns
  const std::array<int, 2>& block_size() const { return _bs; }

private:
  // Maps for the distribution of the ows and columns
  std::array<std::shared_ptr<const common::IndexMap>, 2> _index_maps;

  // Block mode (compact or expanded)
  BlockMode _block_mode;

  // Block sizes
  std::array<int, 2> _bs;

  // Matrix data
  container_type _data;
  column_container_type _cols;
  rowptr_container_type _row_ptr;

  // Start of off-diagonal (unowned columns) on each row
  rowptr_container_type _off_diagonal_offset;

  // Neighborhood communicator (ghost->owner communicator for rows)
  dolfinx::MPI::Comm _comm;

  // -- Precomputed data for finalize/update

  // Request in non-blocking communication
  MPI_Request _request;

  // Position in _data to add received data
  std::vector<int> _unpack_pos;

  // Displacements for alltoall for each neighbor when sending and
  // receiving
  std::vector<int> _val_send_disp, _val_recv_disp;

  // Ownership of each row, by neighbor (for the neighbourhood defined
  // on _comm)
  std::vector<int> _ghost_row_to_rank;

  // Temporary store for finalize data during non-blocking communication
  container_type _ghost_value_data_in;
};

//-----------------------------------------------------------------------------
template <int BS0, int BS1, typename U, typename V, typename W, typename X,
          typename Y>
void impl::set_csr(U&& data, const V& cols, const W& row_ptr, const X& x,
                   const Y& xrows, const Y& xcols,
                   [[maybe_unused]] typename Y::value_type local_size)
{
  const int nc = xcols.size();
  assert(x.size() == xrows.size() * xcols.size() * BS0 * BS1);
  for (std::size_t r = 0; r < xrows.size(); ++r)
  {
    // Row index and current data row
    auto row = xrows[r];
    using T = typename X::value_type;
    const T* xr = x.data() + r * nc * BS0 * BS1;

#ifndef NDEBUG
    if (row >= local_size)
      throw std::runtime_error("Local row out of range");
#endif
    // Columns indices for row
    auto cit0 = std::next(cols.begin(), row_ptr[row]);
    auto cit1 = std::next(cols.begin(), row_ptr[row + 1]);
    for (std::size_t c = 0; c < nc; ++c)
    {
      // Find position of column index
      auto it = std::lower_bound(cit0, cit1, xcols[c]);
      assert(*it == xcols[c]);
      assert(it != cit1);

      std::size_t d = std::distance(cols.begin(), it);
      int di = d * BS0 * BS1;
      int xi = c * BS1;
      assert(di < data.size());
      for (int i = 0; i < BS0; ++i)
      {
        for (int j = 0; j < BS1; ++j)
          data[di + j] = xr[xi + j];
        di += BS1;
        xi += nc * BS1;
      }
    }
  }
}
//-----------------------------------------------------------------------------
// Set with block insertion into a regular CSR (block size 1)
template <int BS0, int BS1, typename U, typename V, typename W, typename X,
          typename Y>
void impl::set_blocked_csr(U&& data, const V& cols, const W& row_ptr,
                           const X& x, const Y& xrows, const Y& xcols,
                           [[maybe_unused]] typename Y::value_type local_size)
{
  const int nc = xcols.size();
  assert(x.size() == xrows.size() * xcols.size() * BS0 * BS1);
  for (std::size_t r = 0; r < xrows.size(); ++r)
  {
    // Row index and current data row
    auto row = xrows[r] * BS0;
    using T = typename X::value_type;
    const T* xr = x.data() + r * nc * BS0 * BS1;

#ifndef NDEBUG
    if (row >= local_size)
      throw std::runtime_error("Local row out of range");
#endif

    for (int i = 0; i < BS0; ++i)
    {
      // Columns indices for row
      auto cit0 = std::next(cols.begin(), row_ptr[row + i]);
      auto cit1 = std::next(cols.begin(), row_ptr[row + i + 1]);
      for (std::size_t c = 0; c < nc; ++c)
      {
        // Find position of column index
        // Assumes the same on all rows of block
        auto it = std::lower_bound(cit0, cit1, xcols[c] * BS1);
        assert(*it == xcols[c] * BS1);
        assert(it != cit1);
        std::size_t d = std::distance(cols.begin(), it);
        assert(d < data.size());
        int xi = c * BS1;
        for (int j = 0; j < BS1; ++j)
          data[d + j] = xr[xi + j];
      }
    }
  }
}
//-----------------------------------------------------------------------------
template <int BS0, int BS1, typename U, typename V, typename W, typename X,
          typename Y>
void impl::add_csr(U&& data, const V& cols, const W& row_ptr, const X& x,
                   const Y& xrows, const Y& xcols)
{
  const int nc = xcols.size();
  assert(x.size() == xrows.size() * xcols.size() * BS0 * BS1);
  for (std::size_t r = 0; r < xrows.size(); ++r)
  {
    // Row index and current data row
    auto row = xrows[r];
    using T = typename X::value_type;
    const T* xr = x.data() + r * nc * BS0 * BS1;

#ifndef NDEBUG
    if (row >= (int)row_ptr.size())
      throw std::runtime_error("Local row out of range");
#endif

    // Columns indices for row
    auto cit0 = std::next(cols.begin(), row_ptr[row]);
    auto cit1 = std::next(cols.begin(), row_ptr[row + 1]);
    for (std::size_t c = 0; c < nc; ++c)
    {
      // Find position of column index
      auto it = std::lower_bound(cit0, cit1, xcols[c]);
      assert(it != cit1);

      std::size_t d = std::distance(cols.begin(), it);
      int di = d * BS0 * BS1;
      int xi = c * BS1;
      assert(di < data.size());
      for (int i = 0; i < BS0; ++i)
      {
        for (int j = 0; j < BS1; ++j)
          data[di + j] += xr[xi + j];
        di += BS1;
        xi += nc * BS1;
      }
    }
  }
}
//-----------------------------------------------------------------------------
template <int BS0, int BS1, typename U, typename V, typename W, typename X,
          typename Y>
void impl::add_blocked_csr(U&& data, const V& cols, const W& row_ptr,
                           const X& x, const Y& xrows, const Y& xcols)
{

  const int nc = xcols.size();
  assert(x.size() == xrows.size() * xcols.size() * BS0 * BS1);
  for (std::size_t r = 0; r < xrows.size(); ++r)
  {
    // Row index and current data row
    auto row = xrows[r] * BS0;
    using T = typename X::value_type;
    const T* xr = x.data() + r * nc * BS0 * BS1;

#ifndef NDEBUG
    if (row >= (int)row_ptr.size())
      throw std::runtime_error("Local row out of range");
#endif

    for (int i = 0; i < BS0; ++i)
    {
      // Columns indices for row
      auto cit0 = std::next(cols.begin(), row_ptr[row + i]);
      auto cit1 = std::next(cols.begin(), row_ptr[row + i + 1]);
      for (std::size_t c = 0; c < nc; ++c)
      {
        // Find position of column index
        auto it = std::lower_bound(cit0, cit1, xcols[c] * BS1);
        assert(*it == xcols[c] * BS1);
        assert(it != cit1);
        std::size_t d = std::distance(cols.begin(), it);
        assert(d < data.size());
        int xi = c * BS1;
        for (int j = 0; j < BS1; ++j)
          data[d + j] += xr[xi + j];
      }
    }
  }
}
//-----------------------------------------------------------------------------
// Add individual entries in block-CSR storage
template <typename U, typename V, typename W, typename X, typename Y>
void impl::add_nonblocked_csr(U&& data, const V& cols, const W& row_ptr,
                              const X& x, const Y& xrows, const Y& xcols,
                              int bs0, int bs1)
{
  const int nc = xcols.size();
  const int nbs = bs0 * bs1;
  typename Y::value_type row, ir;

  assert(x.size() == xrows.size() * xcols.size());
  for (std::size_t r = 0; r < xrows.size(); ++r)
  {
    // Row index and current data row
    row = xrows[r] / bs0;
    ir = xrows[r] % bs0;

    using T = typename X::value_type;
    const T* xr = x.data() + r * nc;

#ifndef NDEBUG
    if (row >= (int)row_ptr.size())
      throw std::runtime_error("Local row out of range");
#endif
    // Columns indices for row
    auto cit0 = std::next(cols.begin(), row_ptr[row]);
    auto cit1 = std::next(cols.begin(), row_ptr[row + 1]);
    for (std::size_t c = 0; c < nc; ++c)
    {
      // Find position of column index
      auto it = std::lower_bound(cit0, cit1, xcols[c] / bs1);
      auto ic = xcols[c] % bs1;
      assert(it != cit1);

      std::size_t d = std::distance(cols.begin(), it);
      int di = d * nbs;
      assert(di < data.size());
      data[di + ir * bs1 + ic] += xr[c];
    }
  }
}
//-----------------------------------------------------------------------------
template <class U, class V, class W, class X>
MatrixCSR<U, V, W, X>::MatrixCSR(const SparsityPattern& p, BlockMode mode)
    : _index_maps({p.index_map(0),
                   std::make_shared<common::IndexMap>(p.column_index_map())}),
      _block_mode(mode), _bs({p.block_size(0), p.block_size(1)}),
      _data(p.num_nonzeros() * _bs[0] * _bs[1], 0),
      _cols(p.graph().first.begin(), p.graph().first.end()),
      _row_ptr(p.graph().second.begin(), p.graph().second.end()),
      _comm(MPI_COMM_NULL)
{
  if (_block_mode == BlockMode::expanded)
  {
    // Rebuild IndexMaps
    for (int i = 0; i < 2; ++i)
    {
      const auto im = _index_maps[i];
      const int size_local = im->size_local() * _bs[i];
      const std::vector<std::int64_t>& ghost_i = im->ghosts();
      std::vector<std::int64_t> ghosts;
      const std::vector<int> ghost_owner_i = im->owners();
      std::vector<int> src_rank;
      for (std::size_t j = 0; j < ghost_i.size(); ++j)
      {
        for (int k = 0; k < _bs[i]; ++k)
        {
          ghosts.push_back(ghost_i[j] * _bs[i] + k);
          src_rank.push_back(ghost_owner_i[j]);
        }
      }
      const std::array<std::vector<int>, 2> src_dest0
          = {_index_maps[i]->src(), _index_maps[i]->dest()};
      _index_maps[i] = std::make_shared<common::IndexMap>(
          _index_maps[i]->comm(), size_local, src_dest0, ghosts, src_rank);
    }

    // Convert sparsity pattern and set _bs to 1

    column_container_type new_cols;
    new_cols.reserve(_data.size());
    rowptr_container_type new_row_ptr = {0};
    new_row_ptr.reserve(_row_ptr.size() * _bs[0]);
    std::span<const std::int32_t> num_diag_nnz = p.off_diagonal_offsets();

    for (auto i = 0; i < _row_ptr.size() - 1; ++i)
    {
      // Repeat row _bs[0] times
      for (int q0 = 0; q0 < _bs[0]; ++q0)
      {
        _off_diagonal_offset.push_back(new_row_ptr.back()
                                       + num_diag_nnz[i] * _bs[1]);
        for (auto j = _row_ptr[i]; j < _row_ptr[i + 1]; ++j)
        {
          for (int q1 = 0; q1 < _bs[1]; ++q1)
            new_cols.push_back(_cols[j] * _bs[1] + q1);
        }
        new_row_ptr.push_back(new_cols.size());
      }
    }
    _cols = new_cols;
    _row_ptr = new_row_ptr;
    _bs[0] = 1;
    _bs[1] = 1;
  }
  else
  {
    // Compute off-diagonal offset for each row (compact)
    std::span<const std::int32_t> num_diag_nnz = p.off_diagonal_offsets();
    _off_diagonal_offset.reserve(num_diag_nnz.size());
    std::transform(num_diag_nnz.begin(), num_diag_nnz.end(), _row_ptr.begin(),
                   std::back_inserter(_off_diagonal_offset), std::plus{});
  }

  // Some short-hand
  const std::array local_size
      = {_index_maps[0]->size_local(), _index_maps[1]->size_local()};
  const std::array local_range
      = {_index_maps[0]->local_range(), _index_maps[1]->local_range()};
  const std::vector<std::int64_t>& ghosts1 = _index_maps[1]->ghosts();

  const std::vector<std::int64_t>& ghosts0 = _index_maps[0]->ghosts();
  const std::vector<int>& src_ranks = _index_maps[0]->src();
  const std::vector<int>& dest_ranks = _index_maps[0]->dest();

  // Create neighbourhood communicator (owner <- ghost)
  MPI_Comm comm;
  MPI_Dist_graph_create_adjacent(_index_maps[0]->comm(), dest_ranks.size(),
                                 dest_ranks.data(), MPI_UNWEIGHTED,
                                 src_ranks.size(), src_ranks.data(),
                                 MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm);
  _comm = dolfinx::MPI::Comm(comm, false);

  // Build map from ghost row index position to owning (neighborhood)
  // rank
  _ghost_row_to_rank.reserve(_index_maps[0]->owners().size());
  for (int r : _index_maps[0]->owners())
  {
    auto it = std::lower_bound(src_ranks.begin(), src_ranks.end(), r);
    assert(it != src_ranks.end() and *it == r);
    int pos = std::distance(src_ranks.begin(), it);
    _ghost_row_to_rank.push_back(pos);
  }

  // Compute size of data to send to each neighbor
  std::vector<std::int32_t> data_per_proc(src_ranks.size(), 0);
  for (std::size_t i = 0; i < _ghost_row_to_rank.size(); ++i)
  {
    assert(_ghost_row_to_rank[i] < data_per_proc.size());
    std::size_t pos = local_size[0] + i;
    data_per_proc[_ghost_row_to_rank[i]] += _row_ptr[pos + 1] - _row_ptr[pos];
  }

  // Compute send displacements
  _val_send_disp.resize(src_ranks.size() + 1, 0);
  std::partial_sum(data_per_proc.begin(), data_per_proc.end(),
                   std::next(_val_send_disp.begin()));

  // For each ghost row, pack and send indices to neighborhood
  std::vector<std::int64_t> ghost_index_data(2 * _val_send_disp.back());
  {
    std::vector<int> insert_pos = _val_send_disp;
    for (std::size_t i = 0; i < _ghost_row_to_rank.size(); ++i)
    {
      const int rank = _ghost_row_to_rank[i];
      int row_id = local_size[0] + i;
      for (int j = _row_ptr[row_id]; j < _row_ptr[row_id + 1]; ++j)
      {
        // Get position in send buffer
        const std::int32_t idx_pos = 2 * insert_pos[rank];

        // Pack send data (row, col) as global indices
        ghost_index_data[idx_pos] = ghosts0[i];
        if (std::int32_t col_local = _cols[j]; col_local < local_size[1])
          ghost_index_data[idx_pos + 1] = col_local + local_range[1][0];
        else
          ghost_index_data[idx_pos + 1] = ghosts1[col_local - local_size[1]];

        insert_pos[rank] += 1;
      }
    }
  }

  // Communicate data with neighborhood
  std::vector<std::int64_t> ghost_index_array;
  std::vector<int> recv_disp;
  {
    std::vector<int> send_sizes;
    std::transform(data_per_proc.begin(), data_per_proc.end(),
                   std::back_inserter(send_sizes),
                   [](auto x) { return 2 * x; });

    std::vector<int> recv_sizes(dest_ranks.size());
    send_sizes.reserve(1);
    recv_sizes.reserve(1);
    MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1,
                          MPI_INT, _comm.comm());

    // Build send/recv displacement
    std::vector<int> send_disp = {0};
    std::partial_sum(send_sizes.begin(), send_sizes.end(),
                     std::back_inserter(send_disp));
    recv_disp = {0};
    std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                     std::back_inserter(recv_disp));

    ghost_index_array.resize(recv_disp.back());
    MPI_Neighbor_alltoallv(ghost_index_data.data(), send_sizes.data(),
                           send_disp.data(), MPI_INT64_T,
                           ghost_index_array.data(), recv_sizes.data(),
                           recv_disp.data(), MPI_INT64_T, _comm.comm());
  }

  // Store receive displacements for future use, when transferring
  // data values
  _val_recv_disp.resize(recv_disp.size());
  const int bs2 = _bs[0] * _bs[1];
  std::transform(recv_disp.begin(), recv_disp.end(), _val_recv_disp.begin(),
                 [&bs2](auto d) { return bs2 * d / 2; });
  std::transform(_val_send_disp.begin(), _val_send_disp.end(),
                 _val_send_disp.begin(), [&bs2](auto d) { return d * bs2; });

  // Global-to-local map for ghost columns
  std::vector<std::pair<std::int64_t, std::int32_t>> global_to_local;
  global_to_local.reserve(ghosts1.size());
  std::int32_t local_i = local_size[1];
  for (std::int64_t idx : ghosts1)
    global_to_local.push_back({idx, global_to_local.size() + local_size[1]});
  std::sort(global_to_local.begin(), global_to_local.end());

  // Compute location in which data for each index should be stored
  // when received
  for (std::size_t i = 0; i < ghost_index_array.size(); i += 2)
  {
    // Row must be on this process
    const std::int32_t local_row = ghost_index_array[i] - local_range[0][0];
    assert(local_row >= 0 and local_row < local_size[0]);

    // Column may be owned or unowned
    std::int32_t local_col = ghost_index_array[i + 1] - local_range[1][0];
    if (local_col < 0 or local_col >= local_size[1])
    {
      auto it = std::lower_bound(global_to_local.begin(), global_to_local.end(),
                                 std::pair(ghost_index_array[i + 1], -1),
                                 [](auto& a, auto& b)
                                 { return a.first < b.first; });
      assert(it != global_to_local.end()
             and it->first == ghost_index_array[i + 1]);
      local_col = it->second;
    }
    auto cit0 = std::next(_cols.begin(), _row_ptr[local_row]);
    auto cit1 = std::next(_cols.begin(), _row_ptr[local_row + 1]);

    // Find position of column index and insert data
    auto cit = std::lower_bound(cit0, cit1, local_col);
    assert(cit != cit1);
    assert(*cit == local_col);
    std::size_t d = std::distance(_cols.begin(), cit);
    _unpack_pos.push_back(d);
  }
}
//-----------------------------------------------------------------------------
template <typename U, typename V, typename W, typename X>
std::vector<typename MatrixCSR<U, V, W, X>::value_type>
MatrixCSR<U, V, W, X>::to_dense() const
{
  const std::size_t nrows = num_all_rows();
  const std::size_t ncols
      = _index_maps[1]->size_local() + _index_maps[1]->num_ghosts();
  std::vector<value_type> A(nrows * ncols * _bs[0] * _bs[1], 0.0);
  for (std::size_t r = 0; r < nrows; ++r)
    for (std::int32_t j = _row_ptr[r]; j < _row_ptr[r + 1]; ++j)
      for (int i0 = 0; i0 < _bs[0]; ++i0)
        for (int i1 = 0; i1 < _bs[1]; ++i1)
        {
          A[(r * _bs[1] + i0) * ncols * _bs[0] + _cols[j] * _bs[1] + i1]
              = _data[j * _bs[0] * _bs[1] + i0 * _bs[1] + i1];
        }

  return A;
}
//-----------------------------------------------------------------------------
template <typename U, typename V, typename W, typename X>
void MatrixCSR<U, V, W, X>::finalize_begin()
{
  const std::int32_t local_size0 = _index_maps[0]->size_local();
  const std::int32_t num_ghosts0 = _index_maps[0]->num_ghosts();
  const int bs2 = _bs[0] * _bs[1];

  // For each ghost row, pack and send values to send to neighborhood
  std::vector<int> insert_pos = _val_send_disp;
  std::vector<value_type> ghost_value_data(_val_send_disp.back());
  for (int i = 0; i < num_ghosts0; ++i)
  {
    const int rank = _ghost_row_to_rank[i];

    // Get position in send buffer to place data to send to this
    // neighbour
    const std::int32_t val_pos = insert_pos[rank];
    std::copy(std::next(_data.data(), _row_ptr[local_size0 + i] * bs2),
              std::next(_data.data(), _row_ptr[local_size0 + i + 1] * bs2),
              std::next(ghost_value_data.begin(), val_pos));
    insert_pos[rank]
        += bs2 * (_row_ptr[local_size0 + i + 1] - _row_ptr[local_size0 + i]);
  }

  _ghost_value_data_in.resize(_val_recv_disp.back());

  // Compute data sizes for send and receive from displacements
  std::vector<int> val_send_count(_val_send_disp.size() - 1);
  std::adjacent_difference(std::next(_val_send_disp.begin()),
                           _val_send_disp.end(), val_send_count.begin());

  std::vector<int> val_recv_count(_val_recv_disp.size() - 1);
  std::adjacent_difference(std::next(_val_recv_disp.begin()),
                           _val_recv_disp.end(), val_recv_count.begin());

  int status = MPI_Ineighbor_alltoallv(
      ghost_value_data.data(), val_send_count.data(), _val_send_disp.data(),
      dolfinx::MPI::mpi_type<value_type>(), _ghost_value_data_in.data(),
      val_recv_count.data(), _val_recv_disp.data(),
      dolfinx::MPI::mpi_type<value_type>(), _comm.comm(), &_request);
  assert(status == MPI_SUCCESS);
}
//-----------------------------------------------------------------------------
template <typename U, typename V, typename W, typename X>
void MatrixCSR<U, V, W, X>::finalize_end()
{
  int status = MPI_Wait(&_request, MPI_STATUS_IGNORE);
  assert(status == MPI_SUCCESS);

  // Add to local rows
  const int bs2 = _bs[0] * _bs[1];
  assert(_ghost_value_data_in.size() == _unpack_pos.size() * bs2);
  for (std::size_t i = 0; i < _unpack_pos.size(); ++i)
    for (int j = 0; j < bs2; ++j)
      _data[_unpack_pos[i] * bs2 + j] += _ghost_value_data_in[i * bs2 + j];

  // Set ghost row data to zero
  const std::int32_t local_size0 = _index_maps[0]->size_local();
  std::fill(std::next(_data.begin(), _row_ptr[local_size0] * bs2), _data.end(),
            0);
}
//-----------------------------------------------------------------------------
template <typename U, typename V, typename W, typename X>
double MatrixCSR<U, V, W, X>::norm_squared() const
{
  const std::size_t num_owned_rows = _index_maps[0]->size_local();
  const int bs2 = _bs[0] * _bs[1];
  assert(num_owned_rows < _row_ptr.size());
  double norm_sq_local = std::accumulate(
      _data.cbegin(), std::next(_data.cbegin(), _row_ptr[num_owned_rows] * bs2),
      double(0), [](auto norm, value_type y) { return norm + std::norm(y); });
  double norm_sq;
  MPI_Allreduce(&norm_sq_local, &norm_sq, 1, MPI_DOUBLE, MPI_SUM, _comm.comm());
  return norm_sq;
}
//-----------------------------------------------------------------------------

} // namespace dolfinx::la
