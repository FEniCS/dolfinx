// Copyright (C) 2007-2009 Garth N. Wells
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2007-2009.
//
// First added:  2007-03-13
// Last changed: 2009-06-19

#ifndef __SPARSITY_PATTERN_H
#define __SPARSITY_PATTERN_H

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

    enum Type {sorted, unsorted};

    /// Create empty sparsity pattern
    SparsityPattern(Type type);

    /// Destructor
    ~SparsityPattern();

    /// Initialize sparsity pattern for a generic tensor
    void init(uint rank, const uint* dims);

    /// Insert non-zero entries
    void insert(const uint* num_rows, const uint * const * rows);

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

    // Sort entries for each row 
    void sort();

    // Return string with statistics for sparsity pattern
    std::string statistics() const;

    // Sparsity pattern type (sorted/unsorted)
    const Type type;    

    // Whether or not pattern has been sorted
    bool _sorted;    

    // Shape of tensor
    std::vector<uint> shape;

    // Local range
    uint range_min;
    uint range_max;

    // Sparsity patterns for diagonal and off-diagonal blocks
    std::vector<std::vector<uint> > diagonal;
    std::vector<std::vector<uint> > off_diagonal;

    // Sparsity pattern for non-local entries stored as [i, j, i, j, ...]
    std::vector<uint> non_local;

  };

}
#endif
