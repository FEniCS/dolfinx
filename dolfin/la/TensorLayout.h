// Copyright (C) 2012 Garth N. Wells
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
// First added:  2012-02-24
// Last changed:

#ifndef __TENSOR_LAYOUT_H
#define __TENSOR_LAYOUT_H

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dolfin/common/types.h"
#include "dolfin/common/MPI.h"

namespace dolfin
{
  class IndexMap;
  class SparsityPattern;

  /// This class described the size and possibly the sparsity of a
  /// (sparse) tensor. It is used by the linear algebra backends to
  /// initialise tensors.

  class TensorLayout : public Variable
  {

  public:

    /// Sparse or dense layout
    enum class Sparsity : bool { SPARSE = true, DENSE = false };

    /// Ghosted or unghosted layout
    enum class Ghosts : bool { GHOSTED = true, UNGHOSTED = false };

    /// Create empty tensor layout
    TensorLayout(std::size_t primary_dim, Sparsity sparsity_pattern);

    /// Create a tensor layout
    TensorLayout(MPI_Comm mpi_comm,
                 std::vector<std::shared_ptr<const IndexMap>> index_maps,
                 std::size_t primary_dim,
                 Sparsity sparsity_pattern,
                 Ghosts ghosted);

    /// Initialize tensor layout
    void init(MPI_Comm mpi_comm,
              std::vector<std::shared_ptr<const IndexMap>> index_maps,
              Ghosts ghosted);

    /// Return rank
    std::size_t rank() const;

    /// Return global size for dimension i (size of tensor, includes
    /// non-zeroes)
    std::size_t size(std::size_t i) const;

    /// Return local range for dimension dim
    std::pair<std::size_t, std::size_t> local_range(std::size_t dim) const;

    /// Return sparsity pattern (possibly null)
    std::shared_ptr<SparsityPattern> sparsity_pattern()
    { return _sparsity_pattern; }

    /// Return sparsity pattern (possibly null), const version
    std::shared_ptr<const SparsityPattern> sparsity_pattern() const
    { return _sparsity_pattern; }

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Primary storage dim (e.g., 0=row major, 1=column major)
    const std::size_t primary_dim;

    /// Return MPI communicator
    MPI_Comm mpi_comm() const
    { return _mpi_comm; }

    /// Return IndexMap for dimension
    std::shared_ptr<const IndexMap> index_map(std::size_t i) const
    {
      dolfin_assert(i < _index_maps.size());
      return _index_maps[i];
    }

    /// Require ghosts
    Ghosts is_ghosted() const
    {
      return _ghosted;
    }

  private:

    // MPI communicator
    MPI_Comm _mpi_comm;

    // Index maps
    std::vector<std::shared_ptr<const IndexMap>> _index_maps;

    // Sparsity pattern
    std::shared_ptr<SparsityPattern> _sparsity_pattern;

    // Ghosted tensor (typically vector) required
    Ghosts _ghosted = Ghosts::UNGHOSTED;

  };

}
#endif
