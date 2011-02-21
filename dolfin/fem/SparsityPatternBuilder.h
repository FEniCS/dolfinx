// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007.
// Modified by Anders Logg, 2008-2009.
//
// First added:  2007-05-24
// Last changed: 2011-02-21

#ifndef __SPARSITY_PATTERN_BUILDER_H
#define __SPARSITY_PATTERN_BUILDER_H

#include <vector>

namespace dolfin
{

  class GenericDofMap;
  class GenericSparsityPattern;
  class Mesh;

  /// This class provides functions to compute the sparsity pattern.

  class SparsityPatternBuilder
  {
  public:

    /// Build sparsity pattern for assembly of given form
    static void build(GenericSparsityPattern& sparsity_pattern,
                      const Mesh& mesh,
                      std::vector<const GenericDofMap*>& dofmaps,
                      bool cells, bool interior_facets);

  };

}

#endif
