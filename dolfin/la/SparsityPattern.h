// Copyright (C) 2007-2011 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/ArrayView.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Set.h>
#include <dolfin/common/types.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dolfin
{

class IndexMap;

/// This class implements a sparsity pattern data structure.  It is
/// used by most linear algebra backends.

class SparsityPattern
{

  // NOTE: Do not change this typedef without performing careful
  //       performance profiling
  /// Set type used for the rows of the sparsity pattern
  typedef dolfin::Set<std::size_t> set_type;

public:
  /// Whether SparsityPattern is sorted
  enum class Type
  {
    sorted,
    unsorted
  };

  /// Ghosted or unghosted layout
  enum class Ghosts : bool
  {
    GHOSTED = true,
    UNGHOSTED = false
  };

  /// Create empty sparsity pattern
  SparsityPattern(MPI_Comm comm, std::size_t primary_dim);

  /// Create a new sparsity pattern by adding sub-patterns, e.g.
  /// pattern =[ pattern00 ][ pattern 01]
  ///          [ pattern10 ][ pattern 11]
  SparsityPattern(
      const std::vector<std::vector<const SparsityPattern*>> patterns);

  /// Create empty sparsity pattern
  SparsityPattern(const SparsityPattern& pattern) = delete;

  /// Destructor
  ~SparsityPattern() {}

  /// Initialize sparsity pattern for a generic tensor
  void init(std::array<std::shared_ptr<const IndexMap>, 2> index_maps,
            Ghosts ghosted);

  /// Insert non-zero entries using global indices
  void insert_global(
      const std::array<ArrayView<const dolfin::la_index_t>, 2>& entries);

  /// Insert non-zero entries using local (process-wise) indices
  void insert_local(
      const std::array<ArrayView<const dolfin::la_index_t>, 2>& entries);

  /// Insert non-zero entries using local (process-wise) indices for
  /// the primary dimension and global indices for the co-dimension
  void insert_local_global(
      const std::array<ArrayView<const dolfin::la_index_t>, 2>& entries);

  /// Insert full rows (or columns, according to primary dimension)
  /// using local (process-wise) indices. This must be called before
  /// any other sparse insertion occurs to avoid quadratic
  /// complexity of dense rows insertion
  void insert_full_rows_local(const std::vector<std::size_t>& rows);

  /// Return primary dimension (e.g., 0=row partition, 1=column
  /// partition)
  std::size_t primary_dim() const { return _primary_dim; }

  /// Return local range for dimension dim
  std::array<std::size_t, 2> local_range(std::size_t dim) const;

  /// Return local range for dimension dim
  std::shared_ptr<const IndexMap> index_map(std::size_t i) const
  {
    dolfin_assert(i < 2);
    return _index_maps[i];
  }

  /// Return number of local nonzeros
  std::size_t num_nonzeros() const;

  /// Fill array with number of nonzeros for diagonal block in
  /// local_range for dimension 0. For matrices, fill array with
  /// number of nonzeros per local row for diagonal block
  void num_nonzeros_diagonal(std::vector<std::size_t>& num_nonzeros) const;

  /// Fill array with number of nonzeros for off-diagonal block in
  /// local_range for dimension 0. For matrices, fill array with
  /// number of nonzeros per local row for off-diagonal block. If
  /// there is no off-diagonal pattern, the vector is resized to
  /// zero-length
  void num_nonzeros_off_diagonal(std::vector<std::size_t>& num_nonzeros) const;

  /// Fill vector with number of nonzeros in local_range for
  /// dimension 0
  void num_local_nonzeros(std::vector<std::size_t>& num_nonzeros) const;

  /// Finalize sparsity pattern
  void apply();

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

  /// Require ghosts
  Ghosts is_ghosted() const { return _ghosted; }

private:
  // Other insertion methods will call this method providing the
  // appropriate mapping of the indices in the entries.
  //
  // The primary dim entries must be local
  // The primary_codim entries must be global
  void insert_entries(
      const std::array<ArrayView<const dolfin::la_index_t>, 2>& entries,
      const std::function<dolfin::la_index_t(const dolfin::la_index_t,
                                             const IndexMap&)>& primary_dim_map,
      const std::function<dolfin::la_index_t(
          const dolfin::la_index_t, const IndexMap&)>& primary_codim_map);

  // Print some useful information
  void info_statistics() const;

  // Primary sparsity pattern storage dimension (e.g., 0=row
  // partition, 1=column partition)
  const std::size_t _primary_dim;

  // MPI communicator
  dolfin::MPI::Comm _mpi_comm;

  // IndexMaps for each dimension
  std::array<std::shared_ptr<const IndexMap>, 2> _index_maps;

  // Sparsity patterns for diagonal and off-diagonal blocks
  std::vector<set_type> _diagonal, _off_diagonal;

  // List of full rows (or columns, according to primary dimension).
  // Full rows are kept separately to circumvent quadratic scaling
  // (caused by linear insertion time into dolfin::Set; std::set has
  // logarithmic insertion, which would result in N log(N) overall
  // complexity for dense rows)
  set_type _full_rows;

  // Cache for non-local entries stored as [i0, j0, i1, j1, ...]. Cleared after
  // communication via apply()
  std::vector<std::size_t> _non_local;

  // Ghosted tensor (typically vector) required
  Ghosts _ghosted = Ghosts::UNGHOSTED;
};
}
