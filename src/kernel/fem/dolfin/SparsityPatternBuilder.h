// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug
//
// First added:  2007-05-24
// Last changed: 2007-12-07

#ifndef __SPARSITY_PATTERN_BUILDER_H
#define __SPARSITY_PATTERN_BUILDER_H

namespace dolfin
{

  class Mesh;
  class GenericSparsityPattern;
  class UFC;

  /// This class provides functions to compute the sparsity pattern.

  class SparsityPatternBuilder
  {
  public:
    
    /// Build sparsity pattern
    static void build(GenericSparsityPattern& sparsity_pattern, Mesh& mesh, UFC& ufc);

  private:

    /// Build scalar sparsity pattern (do nothing)
    static void scalarBuild(GenericSparsityPattern& sparsity_pattern);

    /// Build vector sparsity pattern (compute length of vector)
    static void vectorBuild(GenericSparsityPattern& sparsity_pattern, UFC& ufc);

    /// Build matrix sparsity pattern (compute sparse matrix layput)
    static void matrixBuild(GenericSparsityPattern& sparsity_pattern, 
                                    Mesh& mesh, UFC& ufc);

  };

}

#endif
