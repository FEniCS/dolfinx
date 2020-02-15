// Copyright (C) 2007-2018 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Set.h>
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
  /// @param[in] comm The MPI communicator for the sparsity pattern
  /// @param[in] index_maps The index maps for the rows (0) and columns.
  ///   The index map must contain all indices that will be inserted
  ///   into the sparsity pattern.
  SparsityPattern(
      MPI_Comm comm,
      const std::array<std::shared_ptr<const common::IndexMap>, 2>& index_maps);

  /// Create a new sparsity pattern by adding sub-patterns, e.g.
  /// pattern = [[pattern00, pattern01],
  ///            [pattern10, pattern11]]
  ///
  /// @param[in] comm The MPI communicator for the sparsity pattern
  /// @param[in] patterns The index maps for the rows (0) and columns.
  /// @pre The @p patterns must have been assembled
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

  /// Insert block of non-zero entries using local (process-wise) indices
  /// @param[in] rows The rows indices for the non-zero block
  /// @param[in] cols The column indices for the non-zero block
  void insert(
      const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& rows,
      const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& cols);

  /// Return local range for dimension dim
  /// @param[in] dim Row (0) or column (1) index
  /// @return The local ownership range
  std::array<std::int64_t, 2> local_range(int dim) const;

  /// Index map for dimension dim
  /// @param[in] dim Row (0) or column (1) index
  /// @return The index map
  std::shared_ptr<const common::IndexMap> index_map(int dim) const;

  /// Return number of local nonzeros
  std::size_t num_nonzeros() const;

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

  /// Finalize sparsity pattern and communicate off-process entries
  void assemble();

  /// Return MPI communicator
  MPI_Comm mpi_comm() const { return _mpi_comm.comm(); }

  /// Return informal string representation (pretty-print)
  std::string str() const;

  /// Return underlying sparsity pattern (diagonal)
  const graph::AdjacencyList<std::size_t>& diagonal_pattern() const;

  // /// Return underlying sparsity pattern (off-diagonal)
  // std::vector<std::vector<std::size_t>> off_diagonal_pattern() const;

  /// Print some useful information
  void info_statistics() const;

private:
  // NOTE: Do not change this typedef without performing careful
  //       performance profiling
  // Set type used for the rows of the sparsity pattern
  typedef dolfinx::common::Set<std::size_t> set_type;

  // MPI communicator
  dolfinx::MPI::Comm _mpi_comm;

  // common::IndexMaps for each dimension
  std::array<std::shared_ptr<const common::IndexMap>, 2> _index_maps;

  // Sparsity patterns for diagonal and off-diagonal blocks
  std::vector<set_type> _diagonal_old, _off_diagonal_old;

  // Cache for non-local entries stored as [i0, j0, i1, j1, ...]. i is
  // the local row index and j is the global column index. Cleared after
  // communication via apply().
  std::vector<std::size_t> _non_local;

  std::shared_ptr<graph::AdjacencyList<std::size_t>> _diagonal_new,
      _off_diagonal_new;
};
} // namespace la
} // namespace dolfinx
