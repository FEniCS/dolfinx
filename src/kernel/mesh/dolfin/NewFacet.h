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

  class NewFacet : public MeshEntity
  {
  public:

    /// Constructor
    NewFacet(NewMesh& mesh, uint index) : MeshEntity(mesh, mesh.dim() - 1, index) {}

    /// Destructor
    ~NewFacet() {}

  };

  /// A FacetIterator is a MeshEntityIterator of topological codimension 1.
  
  class NewFacetIterator : public MeshEntityIterator
  {
  public:
    
    NewFacetIterator(NewMesh& mesh) : MeshEntityIterator(mesh, mesh.dim() - 1) {}
    NewFacetIterator(MeshEntity& entity) : MeshEntityIterator(entity, entity.mesh().dim() - 1) {}
    NewFacetIterator(MeshEntityIterator& it) : MeshEntityIterator(it, it->mesh().dim() - 1) {}

  };    

}

#endif
