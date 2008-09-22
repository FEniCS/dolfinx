// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007.
// Modified by Anders Logg, 2008.
//
// First added:  2007-05-24
// Last changed: 2008-01-29

#ifndef __SPARSITY_PATTERN_BUILDER_H
#define __SPARSITY_PATTERN_BUILDER_H

namespace dolfin
{

  class DofMapSet;
  class Mesh;
  class GenericSparsityPattern;
  class UFC;

  /// This class provides functions to compute the sparsity pattern.

  class SparsityPatternBuilder
  {
  public:
    
    /// Build sparsity pattern
    static void build(GenericSparsityPattern& sparsity_pattern, Mesh& mesh,
                      UFC& ufc, const DofMapSet& dof_map_set);

  };

}

#endif
