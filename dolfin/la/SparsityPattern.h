// Copyright (C) 2007-2009 Garth N. Wells
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2007-2009.
//
// First added:  2007-03-13
// Last changed: 2009-06-19

#ifndef __SPARSITY_PATTERN_H
#define __SPARSITY_PATTERN_H

#include <set>
#include <vector>
#include <boost/unordered_set.hpp>
#include "dolfin/common/Set.h"
#include "dolfin/common/types.h"
#include <dolfin/log/LogStream.h>
#include <dolfin/main/MPI.h>
#include "GenericSparsityPattern.h"

namespace dolfin
{

  /// This class implements the GenericSparsityPattern interface.
  /// It is used by most linear algebra backends.

  class SparsityPattern : public GenericSparsityPattern
  {

    // Set type used for the rows of the sparsity pattern
    typedef dolfin::Set<uint> set_type;
    //typedef std::set<uint> set_type;
    //typedef boost::unordered_set<dolfin::uint> set_type;

  public:

    enum Type {sorted, unsorted};

    /// Create empty sparsity pattern
    SparsityPattern();

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

    /// Return total number of nonzeros in local_range for dimension 0
    uint num_nonzeros() const;

    /// Fill array with number of nonzeros for diagonal block in local_range for dimension 0
    /// For matrices, fill array with number of nonzeros per local row for diagonal block
    void num_nonzeros_diagonal(uint* num_nonzeros) const;

    /// Fill array with number of nonzeros for off-diagonal block in local_range for dimension 0
    /// For matrices, fill array with number of nonzeros per local row for off-diagonal block
    void num_nonzeros_off_diagonal(uint* num_nonzeros) const;

    /// Finalize sparsity pattern
    void apply();

    /// Return informal string representation (pretty-print)
    std::string str() const;

    /// Return underlying sparsity pattern (diagonal). Options are
    /// 'sorted' and 'unsorted'.
    std::vector<std::vector<uint> > diagonal_pattern(Type type) const;

    /// Return underlying sparsity pattern (off-diagional). Options are
    /// 'sorted' and 'unsorted'.
    std::vector<std::vector<uint> > off_diagonal_pattern(Type type) const;

  private:

    // Print some useful information
    void info_statistics() const;

    // Shape of tensor
    std::vector<uint> shape;

    // Local range
    uint row_range_min;
    uint row_range_max;
    uint col_range_min;
    uint col_range_max;

    // Sparsity patterns for diagonal and off-diagonal blocks
    std::vector<set_type> diagonal;
    std::vector<set_type> off_diagonal;

    // Sparsity pattern for non-local entries stored as [i, j, i, j, ...]
    std::vector<uint> non_local;

  };

}
#endif
