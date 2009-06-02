// Copyright (C) 2008 Martin Sandve Alnes, Kent-Andre Mardal and Johannes Ring.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2009.
//
// First added:  2008-04-21
// Last changed: 2009-05-23

#ifndef __EPETRA_SPARSITY_PATTERN_H
#define __EPETRA_SPARSITY_PATTERN_H

#ifdef HAS_TRILINOS

#include <dolfin/common/types.h>
#include "GenericSparsityPattern.h"
#include "LinearAlgebraFactory.h"

class Epetra_FECrsGraph;

namespace dolfin
{

  class EpetraVector;
  class EpetraFactory;

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

    /// Sort entries for each row 
    void sort();

    /// Return rank
    uint rank() const;

    /// Return global size for dimension i
    uint size(uint i) const;

    /// Return local range
    std::pair<uint, uint> range() const;

    /// Return total number of nonzeros in local rows
    uint num_nonzeros() const;

    /// Fill array with number of nonzeros per local row for diagonal block
    void num_nonzeros_diagonal(uint* num_nonzeros) const;

    /// Fill array with number of nonzeros per local row for off-diagonal block
    void num_nonzeros_off_diagonal(uint* num_nonzeros) const;

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
