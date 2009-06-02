// Copyright (C) 2007-2008 Garth N. Wells
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2007-2009.
//
// First added:  2007-03-13
// Last changed: 2009-05-23

#ifndef __SPARSITY_PATTERN_H
#define __SPARSITY_PATTERN_H

#include <set>
#include <vector>

#include "GenericSparsityPattern.h"

namespace dolfin
{

  /// This class implements the GenericSparsityPattern interface.
  /// It is used by most linear algebra backends, except for Epetra
  /// which uses a special/native implementation.

  class SparsityPattern: public GenericSparsityPattern
  {
  public:

    /// Create empty sparsity pattern
    SparsityPattern();

    /// Destructor
    ~SparsityPattern();

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

    /// Return informal string representation (pretty-print)
    std::string str() const;
    
    /// Return underlying sparsity pattern
    const std::vector<std::vector<uint> >& pattern() const;

  private:

    // Shape of tensor
    std::vector<uint> shape;

    // Sparsity patterns for diagonal and off-diagonal blocks
    std::vector<std::vector<uint> > diagonal;
    std::vector<std::vector<uint> > off_diagonal;

  };

}
#endif
