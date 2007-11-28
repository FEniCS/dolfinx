// Copyright (C) 2007 Garth N. Wells
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2007.
//
// First added:  2007-03-13
// Last changed: 2007-04-03

#ifndef __SPARSITY_PATTERN_H
#define __SPARSITY_PATTERN_H

#include <set>
#include <vector>

#include <dolfin/dolfin_log.h>
#include <dolfin/constants.h>

namespace dolfin
{

  /// This class represents the sparsity pattern of a vector/matrix. It can be 
  /// used to initalise vectors and sparse matrices. It must be initialised
  /// before use.

  class SparsityPattern
  {
  public:

    /// Constructor
    SparsityPattern();
    
    /// Destructor
    ~SparsityPattern();

    /// Initialise sparsity pattern for a vector
    void init(uint M);

    /// Initialise sparsity pattern for a matrix with total number of rows and columns
    void init(uint M, uint N);

    /// Insert non-zero entry
    void insert(uint row, uint column)
      { sparsity_pattern[row].insert(column); };

    void insert(uint m, const uint* rows, uint n, const uint* cols)
    { 
      for (unsigned int i = 0; i<m;++i)
        for (unsigned int j = 0; i<j;++j)
          sparsity_pattern[rows[i]].insert(cols[j]);
    }

    /// Return global size 
    uint size(uint n) const
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
