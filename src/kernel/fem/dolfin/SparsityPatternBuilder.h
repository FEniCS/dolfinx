// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-05-24
// Last changed: 

#ifndef __SPARSITY_PATTERN_BUILDER_H
#define __SPARSITY_PATTERN_BUILDER_H

namespace dolfin
{

  class DofMapSet;
  class Mesh;
  class SparsityPattern;
  class UFC;

  /// This class provides functions to compute the sparsity pattern.

  class SparsityPatternBuilder
  {
  public:
    
    /// Build sparsity pattern
    static void build(SparsityPattern& sparsity_pattern, Mesh& mesh, UFC& ufc, 
                                    const DofMapSet& dof_map_set);

  private:

    /// Build scalar sparsity pattern (do nothing)
    static void scalarBuild(SparsityPattern& sparsity_pattern);

    /// Build vector sparsity pattern (compute length of vector)
    static void vectorBuild(SparsityPattern& sparsity_pattern,  const DofMapSet& dof_map_set);

    /// Build matrix sparsity pattern (compute sparse matrix layput)
    static void matrixBuild(SparsityPattern& sparsity_pattern, Mesh& mesh, 
                                    UFC& ufc, const DofMapSet& dof_map_set);

  };

}

#endif
