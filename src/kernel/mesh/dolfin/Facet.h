// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-06-02
// Last changed: 2007-05-02

#ifndef __FACET_H
#define __FACET_H

#include <dolfin/MeshEntity.h>
#include <dolfin/MeshEntityIterator.h>

namespace dolfin
{

  /// A Facet is a MeshEntity of topological codimension 1.

  class Facet : public MeshEntity
  {
  public:

    /// Constructor
    Facet(Mesh& mesh, uint index) : MeshEntity(mesh, mesh.topology().dim() - 1, index) {}

    /// Destructor
    ~Facet() {}

  };

  /// A FacetIterator is a MeshEntityIterator of topological codimension 1.
  
  class FacetIterator : public MeshEntityIterator
  {
  public:
    
    FacetIterator(Mesh& mesh) : MeshEntityIterator(mesh, mesh.topology().dim() - 1) {}
    FacetIterator(MeshEntity& entity) : MeshEntityIterator(entity, entity.mesh().topology().dim() - 1) {}

    inline Facet& operator*() { return *operator->(); }
    inline Facet* operator->() { return static_cast<Facet*>(MeshEntityIterator::operator->()); }

  };    

}

#endif
