// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2009.
//
// First added:  2006-06-02
// Last changed: 2009-11-04

#ifndef __FACET_H
#define __FACET_H

#include "Cell.h"
#include "MeshEntity.h"
#include "MeshEntityIterator.h"

namespace dolfin
{

  /// A Facet is a MeshEntity of topological codimension 1.

  class Facet : public MeshEntity
  {
  public:

    /// Constructor
    Facet(const Mesh& mesh, uint index) : MeshEntity(mesh, mesh.topology().dim() - 1, index) {}

    /// Destructor
    ~Facet() {}

    // FIXME: This function should take care of facet 'ownership' when a mesh
    //        is distributed across processes 
    /// Determine whether or not facet is an interior facet. This is 'relative'
    /// to the given partition of the mesh if the mesh is distributed
    bool interior() const
    {
      not_working_in_parallel("Getting adjacent cell");

      if (num_entities(dim() + 1) == 2)
        return true;
      else
        return false;
    }

    // FIXME: This function should take care of the case where adjacent cells
    //        live on different processes
    /// Return adjacent cells
    std::pair<const Cell, const Cell> adjacent_cells() const
    { 
       assert(num_entities(dim() + 1) == 2);
       return std::make_pair(Cell(mesh(), entities(dim() + 1)[0]),
                             Cell(mesh(), entities(dim() + 1)[1]));
    }

  };

  /// A FacetIterator is a MeshEntityIterator of topological codimension 1.

  class FacetIterator : public MeshEntityIterator
  {
  public:

    FacetIterator(const Mesh& mesh) : MeshEntityIterator(mesh, mesh.topology().dim() - 1) {}
    FacetIterator(const MeshEntity& entity) : MeshEntityIterator(entity, entity.mesh().topology().dim() - 1) {}

    inline const Facet& operator*() { return *operator->(); }
    inline const Facet* operator->() { return static_cast<Facet*>(MeshEntityIterator::operator->()); }

  };

}

#endif
