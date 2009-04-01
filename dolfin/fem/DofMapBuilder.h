// Copyright (C) 2008 Anders Logg and Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-08-12
// Last changed: 2009-04-01

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
    static void build(DofMap& dof_map, UFC& ufc, const Mesh& mesh);

  private:
    
    // Build stage 0: Initialize data structures
    static void initialize_data_structure(DofMap& dof_map, const Mesh& mesh);

    // Build stage 1: Compute offsets
    static void compute_offsets();
    
    // Build stage 2: Communicate offsets
    static void communicate_offsets();

    // Build stage 3: Compute dofs that this process is resposible for
    static void number_dofs();
    
    // Build stage 4: Communicate mapping on shared facets
    static void communicate_shared();
    
  };

}

#endif


