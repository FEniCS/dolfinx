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

#include <dolfin/GenericSparsityPattern.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/constants.h>

namespace dolfin
{

  /// This class represents the sparsity pattern of a vector/matrix. It can be 
  /// used to initalise vectors and sparse matrices. It must be initialised
  /// before use.

  class SparsityPattern: public GenericSparsityPattern
  {
  public:

    /// Constructor
    SparsityPattern();
    
    /// Destructor
    ~SparsityPattern();

    /// Initialise sparsity pattern for a matrix with total number of rows and columns
    void init(uint rank, const uint* dims);

    /// Initialise sparsity pattern for a parallel matrix with total number of rows and columns
    void pinit(uint rank, const uint* dims);

    /// Insert non-zero entry
    void insert(const uint* num_rows, const uint * const * rows);

    /// Insert non-zero entry for parallel matrices
    void pinsert(const uint* num_rows, const uint * const * rows);

    /// Return global size 
    uint size(uint n) const;

    /// Return array with number of non-zeroes per row
    void numNonZeroPerRow(uint nzrow[]) const;

    /// Return array with number of non-zeroes per row diagonal and offdiagonal for process_number
    void numNonZeroPerRow(uint process_number, uint d_nzrow[], uint o_nzrow[]) const;

    /// Return total number of non-zeroes
    uint numNonZero() const;

    /// Return underlying sparsity pattern
    const std::vector< std::set<int> >& pattern() const { return sparsity_pattern; };

    /// Display sparsity pattern
    void disp() const;

    void apply() { /* Do nothing */ }

    /// Return array with row range for process_number
    void processRange(uint process_number, uint local_range[]);

    /// Return number of local rows for process_number
    uint numLocalRows(uint process_number) const;


  private:

    /// Initialize range
    void initRange();

    /// Sparsity pattern represented as an vector of sets. Each set corresponds
    /// to a row, and the set contains the column positions of nonzero entries 
    /// When run in parallel this vector contains diagonal non-zeroes
    std::vector< std::set<int> > sparsity_pattern;

    /// Sparsity pattern for off diagonal represented as vector of sets. Each
    /// set corresponds to a row, and the set contains the column positions of nonzero entries 
    std::vector< std::set<int> > o_sparsity_pattern;

    // Dimensions
    uint dim[2];

    //range -array of size + 1 where size is numProcesses + 1. 
    //range[rank], range[rank+1] is the range for processor
    uint* range;
  };
}
#endif
