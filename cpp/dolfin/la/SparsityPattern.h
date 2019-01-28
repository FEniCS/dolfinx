// Copyright (C) 2007-2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Set.h>
#include <memory>
#include <petscsys.h>
#include <string>
#include <utility>
#include <vector>

namespace dolfin
{

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

  // NOTE: Do not change this typedef without performing careful
  //       performance profiling
  /// Set type used for the rows of the sparsity pattern
  typedef dolfin::common::Set<std::size_t> set_type;

public:
  /// Whether SparsityPattern is sorted
  enum class Type
  {
    sorted,
    unsorted
  };

  /// Create an empty sparsity pattern with specified dimensions
  SparsityPattern(
      MPI_Comm comm,
      std::array<std::shared_ptr<const common::IndexMap>, 2> index_maps);

  /// Create a new sparsity pattern by adding sub-patterns, e.g.
  /// pattern =[ pattern00 ][ pattern 01]
  ///          [ pattern10 ][ pattern 11]
  SparsityPattern(
      MPI_Comm comm,
      const std::vector<std::vector<const SparsityPattern*>> patterns);

  SparsityPattern(const SparsityPattern& pattern) = delete;

  /// Move constructor
  SparsityPattern(SparsityPattern&& pattern) = default;

  /// Destructor
  ~SparsityPattern() = default;

  /// Move assignment
  SparsityPattern& operator=(SparsityPattern&& pattern) = default;

  /// Insert non-zero entries using global indices
  void insert_global(
      const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> rows,
      const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> cols);

  /// Insert non-zero entries using local (process-wise) indices
  void insert_local(
      const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> rows,
      const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> cols);

  /// Return local range for dimension dim
  std::array<std::size_t, 2> local_range(std::size_t dim) const;

  /// Return index map for dimension dim
  std::shared_ptr<const common::IndexMap> index_map(std::size_t dim) const;

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
  std::string str(bool verbose) const;

  /// Return underlying sparsity pattern (diagonal). Options are
  /// 'sorted' and 'unsorted'.
  std::vector<std::vector<std::size_t>> diagonal_pattern(Type type) const;

  /// Return underlying sparsity pattern (off-diagonal). Options are
  /// 'sorted' and 'unsorted'. Empty vector is returned if there is
  /// no off-diagonal contribution.
  std::vector<std::vector<std::size_t>> off_diagonal_pattern(Type type) const;

private:
  // Other insertion methods will call this method providing the
  // appropriate mapping of the indices in the entries.
  //
  // The primary dim entries must be local
  // The primary_codim entries must be global
  void insert_entries(
      const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>
          rows,
      const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>
          cols,
      const std::function<PetscInt(const PetscInt, const common::IndexMap&)>&
          row_map,
      const std::function<PetscInt(const PetscInt, const common::IndexMap&)>&
          col_map);

  // Print some useful information
  void info_statistics() const;

  // MPI communicator
  dolfin::MPI::Comm _mpi_comm;

  // common::IndexMaps for each dimension
  std::array<std::shared_ptr<const common::IndexMap>, 2> _index_maps;

  // Sparsity patterns for diagonal and off-diagonal blocks
  std::vector<set_type> _diagonal, _off_diagonal;

  // Cache for non-local entries stored as [i0, j0, i1, j1, ...].
  // Cleared after communication via apply()
  std::vector<std::size_t> _non_local;
};
} // namespace la
} // namespace dolfin