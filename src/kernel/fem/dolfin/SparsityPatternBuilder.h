// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-05-24
// Last changed: 

#ifndef __SPARSITY_PATTERN_BUILDER_H
#define __SPARSITY_PATTERN_BUILDER_H

namespace dolfin
{

  class Mesh;
  class SparsityPattern;
  class UFC;

  /// This class provides functions to compute the sparsity pattern.

  class SparsityPatternBuilder
  {
  public:
    
    /// Build sparsity pattern
    static void build(SparsityPattern& sparsity_pattern, Mesh& mesh, UFC& ufc);

  private:

    static void scalarBuild(SparsityPattern& sparsity_pattern);

    static void vectorBuild(SparsityPattern& sparsity_pattern, UFC& ufc);

    static void matrixBuild(SparsityPattern& sparsity_pattern, Mesh& mesh, 
                                    UFC& ufc);

  };

}

#endif
