// Copyright (C) 2008 Martin Sandve Alnes, Kent-Andre Mardal and Johannes Ring
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg, 2009.
// Modified by Ola Skavhaug, 2009.
//
// First added:  2008-04-21
// Last changed: 2009-08-06

#ifndef __EPETRA_SPARSITY_PATTERN_H
#define __EPETRA_SPARSITY_PATTERN_H

#ifdef HAS_TRILINOS

#include <vector>
#include <dolfin/common/types.h>
#include "GenericSparsityPattern.h"

class Epetra_FECrsGraph;

namespace dolfin
{

  class EpetraVector;

  /// This class implements the GenericSparsityPattern interface for
  /// the Epetra backend. The common interface is mostly
  /// ignored. Instead, the sparsity pattern is represented as an
  /// Epetra_FECrsGraph and a dynamic_cast is used to retrieve the
  /// underlying representation when creating Epetra matrices.

  class EpetraSparsityPattern : public GenericSparsityPattern
  {
  public:

    /// Constructor
    EpetraSparsityPattern();

    /// Destructor
    virtual ~EpetraSparsityPattern();

    /// Initialize sparsity pattern for a generic tensor
    void init(uint rank, const uint* dims);

    /// Insert non-zero entries
    void insert(const uint* num_rows, const uint * const * rows);

    /// Return rank
    uint rank() const;

    /// Return global size for dimension i
    uint size(uint i) const;

    /// Return local range for dimension dim
    std::pair<uint, uint> local_range(uint dim) const;

    /// Return total number of nonzeros in local rows
    uint num_nonzeros() const;

    /// Fill array with number of nonzeros for diagonal block in local_range for dimension 0
    /// For matrices, fill array with number of nonzeros per local row for diagonal block
    void num_nonzeros_diagonal(std::vector<uint>& num_nonzeros) const;

    /// Fill array with number of nonzeros for off-diagonal block in local_range for dimension 0
    /// For matrices, fill array with number of nonzeros per local row for off-diagonal block
    void num_nonzeros_off_diagonal(std::vector<uint>& num_nonzeros) const;

    /// Return underlying sparsity pattern (diagonal). Options are
    /// 'sorted' and 'unsorted'.
    std::vector<std::vector<uint> > diagonal_pattern(Type type) const;

    /// Return underlying sparsity pattern (off-diagional). Options are
    /// 'sorted' and 'unsorted'.
    std::vector<std::vector<uint> > off_diagonal_pattern(Type type) const;

    /// Finalize sparsity pattern
    void apply();

    /// Return Epetra CRS graph
    Epetra_FECrsGraph& pattern() const;

  private:

    // Rank
    uint _rank;

    // Dimensions
    uint dims[2];

    // Epetra representation of sparsity pattern
    Epetra_FECrsGraph* epetra_graph;

  };

}

#endif

#endif
