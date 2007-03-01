// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Hoffman 2006.
//
// First added:  2006-06-01
// Last changed: 2007-03-01

#ifndef __CELL_H
#define __CELL_H

#include <dolfin/Point.h>
#include <dolfin/CellType.h>
#include <dolfin/MeshEntity.h>
#include <dolfin/MeshEntityIterator.h>

namespace dolfin
{

  /// A Cell is a MeshEntity of topological codimension 0.

  class Cell : public MeshEntity
  {
  public:

    /// Constructor
    Cell(Mesh& mesh, uint index) : MeshEntity(mesh, mesh.topology().dim(), index) {}

    /// Destructor
    ~Cell() {}
    
    /// Return type of cell
    inline CellType::Type type() const { return _mesh.type().cellType(); }
    
    /// Return alignment of given entity with respect to the cell
    inline uint alignment(uint dim, uint e) const { return _mesh.type().alignment(*this, dim, e); }

    /// Compute (generalized) volume of cell
    inline real volume() const { return _mesh.type().volume(*this); }

    /// Compute diameter of cell
    inline real diameter() const { return _mesh.type().diameter(*this); }

    /// Compute midpoint of cell
    Point midpoint(); 

    /// Compute component i of normal of given facet with respect to the cell
    inline real normal(uint facet, uint i) const { return _mesh.type().normal(*this, facet, i); }

  };

  /// A CellIterator is a MeshEntityIterator of topological codimension 0.
  
  class CellIterator : public MeshEntityIterator
  {
  public:
    
    CellIterator(Mesh& mesh) : MeshEntityIterator(mesh, mesh.topology().dim()) {}
    CellIterator(MeshEntity& entity) : MeshEntityIterator(entity, entity.mesh().topology().dim()) {}
    CellIterator(MeshEntityIterator& it) : MeshEntityIterator(it, it->mesh().topology().dim()) {}

    inline Cell& operator*() { return *operator->(); }
    inline Cell* operator->() { return static_cast<Cell*>(MeshEntityIterator::operator->()); }

  };    

}

#endif
