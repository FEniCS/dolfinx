// Copyright (C) 2008 Anders Logg and Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-08-12
// Last changed: 2008-08-12

#ifndef __DOF_MAP_BUILDER_H
#define __DOF_MAP_BUILDER_H

namespace dolfin
{
  
  class DofMap;
  class UFC;
  class Mesh;

  /// Documentation of class

  class DofMapBuilder
  {
  public:

    /// Build dof map
    static void build(DofMap& dof_map, UFC& ufc, Mesh& mesh);

  private:

    // Build stage 0: Compute offsets
    static void computeOffsets(uint this_process);
    
    // Build stage 0.5: Communicate offsets
    static void communicateOffsets();

    // Build stage 1: Compute mapping on shared facets
    static void computeShared();
    
    // Build stage 2: Communicate mapping on shared facets
    static void communicateShared();
    
    // Build stage 3: Compute mapping for interior degrees of freedom
    static void computeInterior();

  };

}

#endif


