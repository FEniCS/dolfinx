// Copyright (C) 2006 Johan Hoffman.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-12-20
// Last changed: 2006-12-20

#ifndef __MESHHIERARCHY_H
#define __MESHHIERARCHY_H

#include <dolfin/common/types.h>
#include "MeshFunction.h"

namespace dolfin
{

  class Mesh;

  /// A MeshHierarchy is a set of Mesh objects M_k, each corresponding to 
  /// a refinement level k, for k=0,1,...,k_{max}. 
  /// M_k for k>0 contains mesh entities of codimension 0 that are not 
  /// contained in M_{k-1}, together with associated mesh entities of 
  /// codimension >0.  
  /// 
  /// For example, the MeshHierarchy may correspond to a set of successively 
  /// refined finite element meshes T_k, k=0,1,...,k_{max}, where M_0 
  /// corresponds to cells, nodes and edges of an unrefined initial mesh T_0, 
  /// and M_k corresponds to the cells of the mesh T_k not contained in T_{k-1}, 
  /// together with its nodes and edges. 

  class MeshHierarchy 
  {
  public:
    
    /// Create mesh hierarcy with initial mesh 
    MeshHierarchy(const Mesh& mesh);

    /// Create empty mesh hierarcy
    MeshHierarchy();

    /// Destructor
    ~MeshHierarchy();

    /// Initialize mesh hierarchy 
    void init(const Mesh& mesh); 

    /// Clear mesh hierarchy 
    void clear(); 

    /// Return number of meshes in hierarchy 
    int size();

    /// Add (finest) mesh to mesh hierarchy 
    void add(const Mesh& mesh); 

    /// Remove (finest) mesh from mesh hierarchy 
    void remove(); 

    /// Return reduced mesh object corresponding to level k 
    Mesh& operator() (uint k) const; 

    /// Return full mesh object corresponding to level k 
    Mesh& operator[] (uint k) const; 

  private: 

    /// Array of meshes 
    Mesh* meshes;

    /// Number of meshes in mesh hierarchy 
    uint num_meshes; 

  };

}

#endif


