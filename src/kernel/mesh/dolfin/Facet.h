// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-02
// Last changed: 2006-06-02

#ifndef __NEW_FACET_H
#define __NEW_FACET_H

#include <dolfin/MeshEntity.h>
#include <dolfin/MeshEntityIterator.h>

namespace dolfin
{

  /// A Facet is a MeshEntity of topological codimension 1.

  class Facet : public MeshEntity
  {
  public:

    /// Constructor
    Facet(Mesh& mesh, uint index) : MeshEntity(mesh, mesh.dim() - 1, index) {}

    /// Destructor
    ~Facet() {}

  };

  /// A FacetIterator is a MeshEntityIterator of topological codimension 1.
  
  class FacetIterator : public MeshEntityIterator
  {
  public:
    
    FacetIterator(Mesh& mesh) : MeshEntityIterator(mesh, mesh.dim() - 1) {}
    FacetIterator(MeshEntity& entity) : MeshEntityIterator(entity, entity.mesh().dim() - 1) {}
    FacetIterator(MeshEntityIterator& it) : MeshEntityIterator(it, it->mesh().dim() - 1) {}

    inline Facet& operator*()
    { return static_cast<Facet&>(*static_cast<MeshEntityIterator>(*this)); }

    inline Facet* operator->()
    { return &static_cast<Facet&>(*static_cast<MeshEntityIterator>(*this)); }

  };    

}

#endif
