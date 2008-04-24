// Copyright (C) 2008 Martin Sandve Alnes, Kent-Andre Mardal and Johannes Ring.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-04-21
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
  /// Base class for sparsity patterns of vectors/matrices. Concrete sub classes can 
  /// be used to initalise vectors and sparse matrices.
  class EpetraSparsityPattern : public GenericSparsityPattern 
  {
  public:

    /// Constructor
    EpetraSparsityPattern(); 

    /// Destructor
    virtual ~EpetraSparsityPattern();

    /// Initialise sparsity pattern for a generic tensor
    void init(uint rank, const uint* dims);

    /// Initialise sparsity pattern for a parallel generic tensor
    void pinit(uint rank, const uint* dims);

    /// Insert non-zero entry
    void insert(const uint* num_rows, const uint * const * rows);

    /// Insert non-zero entry
    void pinsert(const uint* num_rows, const uint * const * rows);

    /// Return global size 
    uint size(uint n) const;

    /// Return array with number of non-zeroes per row
    void numNonZeroPerRow(uint nzrow[]) const;

    /// Return total number of non-zeroes
    uint numNonZero() const;

    /// Finalize sparsity pattern (needed by most parallel la backends)
    void apply();

    /// Return factory object for backend
    LinearAlgebraFactory& factory() const;

    Epetra_FECrsGraph& pattern() const;  

  private: 
    Epetra_FECrsGraph* epetra_graph; 
    uint                rank; 
    uint*               dims; 

  };

}
#endif
#endif


