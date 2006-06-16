// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-01
// Last changed: 2006-06-01

#ifndef __NEW_CELL_H
#define __NEW_CELL_H

#include <dolfin/MeshEntity.h>
#include <dolfin/MeshEntityIterator.h>

namespace dolfin
{

  /// A Cell is a MeshEntity of topological codimension 0.

  class NewCell : public MeshEntity
  {
  public:

    /// Constructor
    NewCell(NewMesh& mesh, uint index) : MeshEntity(mesh, mesh.dim(), index) {}

    /// Destructor
    ~NewCell() {}

  };

  /// A CellIterator is a MeshEntityIterator of topological codimension 0.
  
  class NewCellIterator : public MeshEntityIterator
  {
  public:
    
    NewCellIterator(NewMesh& mesh) : MeshEntityIterator(mesh, mesh.dim()) {}
    NewCellIterator(MeshEntity& entity) : MeshEntityIterator(entity, entity.mesh().dim()) {}
    NewCellIterator(MeshEntityIterator& it) : MeshEntityIterator(it, it->mesh().dim()) {}

    inline NewCell& operator*()
    { return static_cast<NewCell&>(*static_cast<MeshEntityIterator>(*this)); }

    inline NewCell* operator->()
    { return &static_cast<NewCell&>(*static_cast<MeshEntityIterator>(*this)); }

  };    

}

#endif
