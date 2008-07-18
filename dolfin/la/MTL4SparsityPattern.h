// Copyright (C) 2008 Dag Lindbo
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-07-06
// Last changed: 2008-07-06

#ifndef __MTL4_SPARSITY_PATTERN_H
#define __MTL4_SPARSITY_PATTERN_H

#ifdef HAS_MTL4

#include <dolfin/common/types.h>
#include "GenericSparsityPattern.h"
#include "LinearAlgebraFactory.h"

namespace dolfin
{
  class MTL4Vector; 
  class MTL4Factory; 

  class MTL4SparsityPattern : public GenericSparsityPattern 
  {
  public:

    /// Constructor
    MTL4SparsityPattern(); 

    /// Destructor
    virtual ~MTL4SparsityPattern();

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

    //    MTL4_FECrsGraph& pattern() const;  

  private: 
    uint                rank; 
    uint*               dims; 

  };

}
#endif
#endif


