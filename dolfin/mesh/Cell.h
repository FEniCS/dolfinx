// Copyright (C) 2006-2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Hoffman 2006.
// Modified by Andre Massing 2009.
//
// First added:  2006-06-01
// Last changed: 2010-08-16

#ifndef __CELL_H
#define __CELL_H

#include "Point.h"
#include "CellType.h"
#include "MeshEntity.h"
#include "MeshEntityIterator.h"
#include "MeshFunction.h"

namespace dolfin
{

  /// A Cell is a MeshEntity of topological codimension 0.

  class Cell : public MeshEntity
  {
  public:

    /// Create empty cell
    Cell() : MeshEntity() {}

    /// Create cell on given mesh with given index
    Cell(const Mesh& mesh, uint index) : MeshEntity(mesh, mesh.topology().dim(), index) {}

    /// Destructor
    ~Cell() {}

    /// Return type of cell
    inline CellType::Type type() const { return _mesh->type().cell_type(); }

    /// Compute orientation of cell (0 is right, 1 is left)
    inline double orientation() const
    { return _mesh->type().orientation(*this); }

    /// Compute (generalized) volume of cell
    inline double volume() const
    { return _mesh->type().volume(*this); }

    /// Compute diameter of cell
    inline double diameter() const
    { return _mesh->type().diameter(*this); }

    /// Compute component i of normal of given facet with respect to the cell
    inline double normal(uint facet, uint i) const
    { return _mesh->type().normal(*this, facet, i); }

    /// Compute normal of given facet with respect to the cell
    inline Point normal(uint facet) const
    { return _mesh->type().normal(*this, facet); }

    /// Compute the area/length of given facet with respect to the cell
    inline double facet_area(uint facet) const
    { return _mesh->type().facet_area(*this, facet); }

    /// Order entities locally
    inline void order(MeshFunction<uint>* global_vertex_indices)
    { _mesh->type().order(*this, global_vertex_indices); }

    /// Check if entities are ordered
    inline bool ordered(MeshFunction<uint>* global_vertex_indices)
    { return _mesh->type().ordered(*this, global_vertex_indices); }

  };

  /// A CellIterator is a MeshEntityIterator of topological codimension 0.
  class CellIterator : public MeshEntityIterator
  {
  public:

    CellIterator() : MeshEntityIterator() {}
    CellIterator(const Mesh& mesh) : MeshEntityIterator(mesh, mesh.topology().dim()) {}
    CellIterator(const MeshEntity& entity) : MeshEntityIterator(entity, entity.mesh().topology().dim()) {}

    inline Cell& operator*() { return *operator->(); }
    inline Cell* operator->() { return static_cast<Cell*>(MeshEntityIterator::operator->()); }

  };

  /// A CellFunction is a MeshFunction of topological codimension 0.

  template <class T> class CellFunction : public MeshFunction<T>
  {
  public:

    CellFunction(const Mesh& mesh) : MeshFunction<T>(mesh, mesh.topology().dim()) {}

  };

}

#endif
