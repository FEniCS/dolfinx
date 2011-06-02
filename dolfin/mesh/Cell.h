// Copyright (C) 2006-2010 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Johan Hoffman 2006.
// Modified by Andre Massing 2009.
// Modified by Garth N. Wells 2010.
//
// First added:  2006-06-01
// Last changed: 2010-12-06

#ifndef __CELL_H
#define __CELL_H

#include "CellType.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshEntityIterator.h"
#include "MeshFunction.h"
#include "Point.h"

namespace dolfin
{

  /// A Cell is a MeshEntity of topological codimension 0.

  class Cell : public MeshEntity
  {
  public:

    /// Create empty cell
    Cell() : MeshEntity() {}

    /// Create cell on given mesh with given index
    Cell(const Mesh& mesh, uint index)
      : MeshEntity(mesh, mesh.topology().dim(), index) {}

    /// Destructor
    ~Cell() {}

    /// Return type of cell
    CellType::Type type() const
    { return _mesh->type().cell_type(); }

    /// Compute orientation of cell (0 is right, 1 is left)
    double orientation() const
    { return _mesh->type().orientation(*this); }

    /// Compute (generalized) volume of cell
    double volume() const
    { return _mesh->type().volume(*this); }

    /// Compute diameter of cell
    double diameter() const
    { return _mesh->type().diameter(*this); }

    /// Compute component i of normal of given facet with respect to the cell
    double normal(uint facet, uint i) const
    { return _mesh->type().normal(*this, facet, i); }

    /// Compute normal of given facet with respect to the cell
    Point normal(uint facet) const
    { return _mesh->type().normal(*this, facet); }

    /// Compute the area/length of given facet with respect to the cell
    double facet_area(uint facet) const
    { return _mesh->type().facet_area(*this, facet); }

    /// Order entities locally
    void order(const MeshFunction<uint>* global_vertex_indices)
    { _mesh->type().order(*this, global_vertex_indices); }

    /// Check if entities are ordered
    bool ordered(const MeshFunction<uint>* global_vertex_indices) const
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

    CellFunction(const Mesh& mesh)
      : MeshFunction<T>(mesh, mesh.topology().dim()) {}

    CellFunction(const Mesh& mesh, const T& value)
      : MeshFunction<T>(mesh, mesh.topology().dim(), value) {}

  };

}

#endif
