// Copyright (C) 2007-2011 Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg, 2007-2009.
//
// First added:  2007-03-13
// Last changed: 2011-01-02

#ifndef __SPARSITY_PATTERN_H
#define __SPARSITY_PATTERN_H

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <dolfin/common/ArrayView.h>
#include <dolfin/common/Set.h>
#include <dolfin/common/types.h>

namespace dolfin
{

  class IndexMap;

  /// This class implements a sparsity pattern data structure.  It is
  /// used by most linear algebra backends.

  class SparsityPattern
  {

    // NOTE: Do not change this typedef without performing careful
    //       performance profiling
    // Set type used for the rows of the sparsity pattern
    typedef dolfin::Set<std::size_t> set_type;

  public:

    enum class Type {sorted, unsorted};

    /// Create empty sparsity pattern
    SparsityPattern(std::size_t primary_dim);

    /// Create sparsity pattern for a generic tensor
    SparsityPattern(MPI_Comm mpi_comm,
                    std::vector<std::shared_ptr<const IndexMap>> index_maps,
                    std::size_t primary_dim);

    /// Initialize sparsity pattern for a generic tensor
    void init(MPI_Comm mpi_comm,
              std::vector<std::shared_ptr<const IndexMap>> index_maps);

    /// Insert a global entry - will be fixed by apply()
    void insert_global(dolfin::la_index i, dolfin::la_index j);

    /// Insert non-zero entries using global indices
    void insert_global(const std::vector<
                       ArrayView<const dolfin::la_index>>& entries);

    /// Insert non-zero entries using local (process-wise) indices
    void insert_local(const std::vector<
                      ArrayView<const dolfin::la_index>>& entries);

    /// Return rank
    std::size_t rank() const;

    /// Return primary dimension (e.g., 0=row partition, 1=column
    /// partition)
    std::size_t primary_dim() const
    { return _primary_dim; }

    /// Return local range for dimension dim
    std::pair<std::size_t, std::size_t> local_range(std::size_t dim) const;

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
    void
      num_nonzeros_off_diagonal(std::vector<std::size_t>& num_nonzeros) const;

    /// Fill vector with number of nonzeros in local_range for
    /// dimension 0
    void num_local_nonzeros(std::vector<std::size_t>& num_nonzeros) const;

    /// Finalize sparsity pattern
    void apply();

    // Return MPI communicator
    MPI_Comm mpi_comm() const
    { return _mpi_comm; }

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Return underlying sparsity pattern (diagonal). Options are
    /// 'sorted' and 'unsorted'.
    std::vector<std::vector<std::size_t>> diagonal_pattern(Type type) const;

    /// Return underlying sparsity pattern (off-diagonal). Options are
    /// 'sorted' and 'unsorted'.
    std::vector<std::vector<std::size_t>> off_diagonal_pattern(Type type) const;

  private:

    // Print some useful information
    void info_statistics() const;

    // Primary sparsity pattern storage dimension (e.g., 0=row
    // partition, 1=column partition)
    const std::size_t _primary_dim;

   // MPI communicator
    MPI_Comm _mpi_comm;

    // Ownership range for each dimension
    //    std::vector<std::pair<std::size_t, std::size_t>> _local_range;

    // IndexMaps for each dimension
    std::vector<std::shared_ptr<const IndexMap>> _index_maps;

    // Sparsity patterns for diagonal and off-diagonal blocks
    std::vector<set_type> diagonal;
    std::vector<set_type> off_diagonal;

    // Sparsity pattern for non-local entries stored as [i0, j0, i1, j1, ...]
    std::vector<std::size_t> non_local;

  };

}
#endif
