// Copyright (C) 2006 Johan Hoffman.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-12-20
// Last changed: 2006-12-20

#ifndef __MESHHIERARCHY_H
#define __MESHHIERARCHY_H

#include <dolfin/Mesh.h>

namespace dolfin
{

  /// A MeshHierarchy is a set of Mesh objects M_k, each corresponding to 
  /// a refinement level k, for k=0,1,...,k_{max}. 
  /// M_k for k>0 contains mesh entities of codimension 0 that are not 
  /// contained in M_{k-1}, together with associated mesh entities of 
  /// codimension > 0.  
  /// 
  /// For example, the MeshHierarchy may correspond to a set of successively 
  /// refined finite element meshes T_k, k=0,1,...,k_{max}, where M_0 
  /// corresponds to cells, nodes and edges of an unrefined initial mesh T_0, 
  /// and M_k corresponds to the cells of the mesh T_k not contained in T_{k-1}, 
  /// together with its nodes and edges. 

  class MeshHierarchy 
  {
  public:
    
    /// Create empty MeshHierarcy
    MeshHierarchy();

    /// Destructor
    ~MeshHierarchy();

    /// Return full Mesh object corresponding to MeshHierarchy level k.  
    Mesh operator[] (uint k) const; 

    /// Return reduced Mesh object corresponding to MeshHierarchy level k.  
    Mesh operator() (uint k) const; 

  };

}

#endif


