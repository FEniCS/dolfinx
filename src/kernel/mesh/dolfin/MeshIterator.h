// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __MESH_ITERATOR_H
#define __MESH_ITERATOR_H

#include <dolfin/Array.h>
#include <dolfin/General.h>

namespace dolfin {
  
  class Mesh;
  class MeshHierarchy;  
  
  typedef Mesh* MeshPointer;
  
  /// Iterator for the meshs in a mesh hierarchy.

  class MeshIterator {
  public:
    
    /// Create an iterator positioned at the top (coarsest) mesh
    MeshIterator(const MeshHierarchy& meshs);

    /// Create an iterator positioned at the given position
    MeshIterator(const MeshHierarchy& meshs, Index index);

    /// Destructor
    ~MeshIterator();
   
    /// Step to next mesh
    MeshIterator& operator++();

    /// Step to previous mesh
    MeshIterator& operator--();

    /// Check if iterator has reached the first (or last) mesh
    bool end();

    /// Return index for current position
    int index();
	 
    operator MeshPointer() const;
    Mesh& operator*() const;
    Mesh* operator->() const;

  private:
    
    Array<Mesh*>::Iterator it;
	
  };

}

#endif
