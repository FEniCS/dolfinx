// Copyright (C) 2007 Garth N. Wells
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-03-13
// Last changed: 2007-03-15

#ifndef __SPARSITY_PATTERN_H
#define __SPARSITY_PATTERN_H

#include <set>
#include <vector>
#include <dolfin/constants.h>

namespace dolfin
{

  /// This class represents the sparsity pattern of a matrix. It will be used
  /// to initalise sparse matrices.

  class SparsityPattern
  {
  public:

    /// Constructor
    SparsityPattern();
    
    /// Destructor
    ~SparsityPattern();

    /// Initialise sparsity pattern with total number of rows and columns
    void init(const uint M, const uint N);

    /// Insert non-zero entry
    void insert(const uint row, const uint column)
      { sparsity_pattern[row].insert(column); };

    /// Return global size 
    uint size(const uint n) const
    { 
      dolfin_assert(n < 2);
      return dim[n]; 
    };

    /// Return array with number of non-zeroes per row
    void numNonZeroPerRow(uint nzrow[]) const;

    /// Return total number of non-zeroes
    uint numNonZero() const;

    /// Return underlying sparsity pattern
    const std::vector< std::set<int> >& pattern() const { return sparsity_pattern; };

    /// Display sparsity pattern
    void disp() const;

  private:

    /// Sparsity pattern represented as an vector of sets. Each set corresponds
    /// to a row, and the set contains the column positions of nonzero entries 
    std::vector< std::set<int> > sparsity_pattern;

    // Dimensions
    uint dim[2];

  };
}
#endif
