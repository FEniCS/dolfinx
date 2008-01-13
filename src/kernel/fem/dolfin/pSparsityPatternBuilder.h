// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug
//
// First added:  2007-05-24
// Last changed: 2007-12-07

#ifndef __P_SPARSITY_PATTERN_BUILDER_H
#define __P_SPARSITY_PATTERN_BUILDER_H

namespace dolfin
{

  class pDofMapSet;
  class Mesh;
  class GenericSparsityPattern;
  class pUFC;

  /// This class provides functions to compute the sparsity pattern.

  class pSparsityPatternBuilder
  {
  public:
    
    /// Build sparsity pattern
    static void build(GenericSparsityPattern& sparsity_pattern, Mesh& mesh, pUFC& ufc, 
                                    const pDofMapSet& dof_map_set);

  private:

    /// Build scalar sparsity pattern (do nothing)
    static void scalarBuild(GenericSparsityPattern& sparsity_pattern);

    /// Build vector sparsity pattern (compute length of vector)
    static void vectorBuild(GenericSparsityPattern& sparsity_pattern,  const pDofMapSet& dof_map_set);

    /// Build matrix sparsity pattern (compute sparse matrix layput)
    static void matrixBuild(GenericSparsityPattern& sparsity_pattern, Mesh& mesh, 
                                    pUFC& ufc, const pDofMapSet& dof_map_set);

  };

}

#endif
