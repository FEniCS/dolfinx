// Copyright (C) 2006-2013 Anders Logg
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
// Last changed: 2013-01-09

#ifndef __CELL_H
#define __CELL_H

#include <ufc.h>

#include "CellType.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshEntityIteratorBase.h"
#include "MeshFunction.h"
#include "Point.h"

namespace dolfin
{

  /// A Cell is a _MeshEntity_ of topological codimension 0.

  class Cell : public MeshEntity,
               public ufc::cell_topology,
               public ufc::cell_geometry
  {
  public:

    /// Create empty cell
    Cell() : MeshEntity() {}

    /// Create cell on given mesh with given index
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh.
    ///     index (std::size_t)
    ///         The index.
    Cell(const Mesh& mesh, std::size_t index)
      : MeshEntity(mesh, mesh.topology().dim(), index) {}

    /// Destructor
    ~Cell() {}

    /// Return type of cell
    CellType::Type type() const
    { return _mesh->type().cell_type(); }

    /// Compute orientation of cell
    ///
    /// *Returns*
    ///     std::size_t
    ///         Orientation of the cell (0 is 'up'/'right', 1 is 'down'/'left')
    std::size_t orientation() const
    { return _mesh->type().orientation(*this); }

    /// Compute orientation of cell relative to given 'up' direction
    ///
    /// *Arguments*
    ///     up (_Point_)
    ///         The direction defined as 'up'
    ///
    /// *Returns*
    ///     std::size_t
    ///         Orientation of the cell (0 is 'same', 1 is 'opposite')
    std::size_t orientation(const Point& up) const
    { return _mesh->type().orientation(*this, up); }

    /// Compute (generalized) volume of cell
    ///
    /// *Returns*
    ///     double
    ///         The volume of the cell.
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         UnitSquare mesh(1, 1);
    ///         Cell cell(mesh, 0);
    ///         info("%g", cell.volume());
    ///
    ///     output::
    ///
    ///         0.5
    double volume() const
    { return _mesh->type().volume(*this); }

    /// Compute diameter of cell
    ///
    /// *Returns*
    ///     double
    ///         The diameter of the cell.
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         UnitSquare mesh(1, 1);
    ///         Cell cell(mesh, 0);
    ///         info("%g", cell.diameter());
    ///
    ///     output::
    ///
    ///         1.41421
    double diameter() const
    { return _mesh->type().diameter(*this); }

    /// Compute component i of normal of given facet with respect to the cell
    ///
    /// *Arguments*
    ///     facet (std::size_t)
    ///         Index of facet.
    ///     i (std::size_t)
    ///         Component.
    ///
    /// *Returns*
    ///     double
    ///         Component i of the normal of the facet.
    double normal(std::size_t facet, std::size_t i) const
    { return _mesh->type().normal(*this, facet, i); }

    /// Compute normal of given facet with respect to the cell
    ///
    /// *Arguments*
    ///     facet (std::size_t)
    ///         Index of facet.
    ///
    /// *Returns*
    ///     _Point_
    ///         Normal of the facet.
    Point normal(std::size_t facet) const
    { return _mesh->type().normal(*this, facet); }

    /// Compute normal to cell itself (viewed as embedded in 3D)
    ///
    /// *Returns*
    ///     _Point_
    ///         Normal of the cell
    Point cell_normal() const
    { return _mesh->type().cell_normal(*this); }

    /// Compute the area/length of given facet with respect to the cell
    ///
    /// *Arguments*
    ///     facet (std::size_t)
    ///         Index of the facet.
    ///
    /// *Returns*
    ///     double
    ///         Area/length of the facet.
    double facet_area(std::size_t facet) const
    { return _mesh->type().facet_area(*this, facet); }

    /// Order entities locally
    ///
    /// *Arguments*
    ///     global_vertex_indices (_MeshFunction_ <std::size_t>)
    ///         The global vertex indices.
    void order(const std::vector<std::size_t>& local_to_global_vertex_indices)
    { _mesh->type().order(*this, local_to_global_vertex_indices); }

    /// Check if entities are ordered
    ///
    /// *Arguments*
    ///     global_vertex_indices (_MeshFunction_ <std::size_t>)
    ///         The global vertex indices.
    ///
    /// *Returns*
    ///     bool
    ///         True if ordered.
    bool ordered(const std::vector<std::size_t>& local_to_global_vertex_indices) const
    { return _mesh->type().ordered(*this, local_to_global_vertex_indices); }

    //--- Implementation of the UFC cell_topology interface ---

    /// Return array of global entity indices for topological dimension d
    const std::size_t* entity_indices(std::size_t d) const
    {
      return _mesh->topology()(_dim, 0)(_local_index);
    }

    //--- Implementation of the UFC cell_geometry interface ---

    /// Return array of coordinates for vertex i
    const double* vertex_coordinates(std::size_t i) const
    {
      return _mesh->geometry().x(_mesh->topology()(_dim, 0)(_local_index)[i]);
    }

  };

  /// A CellIterator is a MeshEntityIterator of topological codimension 0.
  typedef MeshEntityIteratorBase<Cell> CellIterator;

  /// A CellFunction is a MeshFunction of topological codimension 0.
  template <typename T> class CellFunction : public MeshFunction<T>
  {
  public:

    CellFunction(const Mesh& mesh)
      : MeshFunction<T>(mesh, mesh.topology().dim()) {}

    CellFunction(const Mesh& mesh, const T& value)
      : MeshFunction<T>(mesh, mesh.topology().dim(), value) {}

  };

}

#endif
