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

  class Cell : public MeshEntity
  {
  public:

    /// Constructor
    Cell(Mesh& mesh, uint index) : MeshEntity(mesh, mesh.dim(), index) {}

    /// Destructor
    ~Cell() {}

  };

  /// A CellIterator is a MeshEntityIterator of topological codimension 0.
  
  class CellIterator : public MeshEntityIterator
  {
  public:
    
    CellIterator(Mesh& mesh) : MeshEntityIterator(mesh, mesh.dim()) {}
    CellIterator(MeshEntity& entity) : MeshEntityIterator(entity, entity.mesh().dim()) {}
    CellIterator(MeshEntityIterator& it) : MeshEntityIterator(it, it->mesh().dim()) {}

    inline Cell& operator*()
    { return static_cast<Cell&>(*static_cast<MeshEntityIterator>(*this)); }

    inline Cell* operator->()
    { return &static_cast<Cell&>(*static_cast<MeshEntityIterator>(*this)); }

  };    

}

#endif
