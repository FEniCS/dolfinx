// Copyright (C) 2007-2020 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfinx/common/MPI.h>
#include <memory>
#include <petscsys.h>
#include <string>
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
      const std::array<std::shared_ptr<const common::IndexMap>, 2>& index_maps);

  /// Create a new sparsity pattern by adding sub-patterns, e.g.
  /// pattern =[ pattern00 ][ pattern 01]
  ///          [ pattern10 ][ pattern 11]
  SparsityPattern(
      MPI_Comm comm,
      const std::vector<std::vector<const SparsityPattern*>>& patterns);

  SparsityPattern(const SparsityPattern& pattern) = delete;

  /// Move constructor
  SparsityPattern(SparsityPattern&& pattern) = default;

  /// Destructor
  ~SparsityPattern() = default;

  /// Move assignment
  SparsityPattern& operator=(SparsityPattern&& pattern) = default;

  /// Return local range for dimension dim
  std::array<std::int64_t, 2> local_range(int dim) const;

  /// Return index map for dimension dim
  std::shared_ptr<const common::IndexMap> index_map(int dim) const;

  /// Insert non-zero locations using local (process-wise) indices
  void insert(
      const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& rows,
      const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& cols);

  /// Insert non-zero locations on the diagonal
  /// @param[in] rows The rows in local (process-wise) indices. The
  ///   indices must exist in the row IndexMap.
  void insert_diagonal(
      const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& rows);

  /// Finalize sparsity pattern and communicate off-process entries
  void assemble();

  /// Return number of local nonzeros
  std::int64_t num_nonzeros() const;

  /// Fill array with number of nonzeros per row for diagonal block in
  /// local_range for dimension 0
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> num_nonzeros_diagonal() const;

  /// Fill array with number of nonzeros for off-diagonal block in
  /// local_range for dimension 0. If there is no off-diagonal pattern,
  /// the returned vector will have zero-length.
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1>
  num_nonzeros_off_diagonal() const;

  /// Fill vector with number of nonzeros in local_range for
  /// dimension 0
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> num_local_nonzeros() const;

  /// Sparsity pattern for the owned (diagonal) block. Uses local
  /// indices for the columns.
  const graph::AdjacencyList<std::int32_t>& diagonal_pattern() const;

  /// Sparsity pattern for the un-owned (off-diagonal) columns. Uses global
  /// indices for the columns.
  const graph::AdjacencyList<std::int64_t>& off_diagonal_pattern() const;

  /// Return MPI communicator
  MPI_Comm mpi_comm() const;

  /// Return informal string representation (pretty-print)
  std::string str() const;

private:
  // MPI communicator
  dolfinx::MPI::Comm _mpi_comm;

  // common::IndexMaps for each dimension
  std::array<std::shared_ptr<const common::IndexMap>, 2> _index_maps;

  // Caches for diagonal and off-diagonal blocks
  std::vector<std::vector<std::int32_t>> _diagonal_cache;
  std::vector<std::vector<std::int64_t>> _off_diagonal_cache;

  std::shared_ptr<graph::AdjacencyList<std::int32_t>> _diagonal;
  std::shared_ptr<graph::AdjacencyList<std::int64_t>> _off_diagonal;
};
} // namespace la
} // namespace dolfinx
