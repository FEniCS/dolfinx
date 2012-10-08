// Copyright (C) 2006-2009 Anders Logg
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
// Modified by Kristoffer Selim, 2008.
// Modified by Andre Massing, 2009-2010.
//
// First added:  2006-06-05
// Last changed: 2011-03-17

#ifndef __CELL_TYPE_H
#define __CELL_TYPE_H

#include <string>
#include <vector>
#include <dolfin/common/types.h>

namespace dolfin
{

  class Cell;
  class MeshEditor;
  class MeshEntity;
  template <typename T> class MeshFunction;
  class Point;

  /// This class provides a common interface for different cell types.
  /// Each cell type implements mesh functionality that is specific to
  /// a certain type of cell.

  class CellType
  {
  public:

    /// Enum for different cell types
    enum Type { point, interval, triangle, tetrahedron };

    /// Constructor
    CellType(Type cell_type, Type facet_type);

    /// Destructor
    virtual ~CellType();

    /// Create cell type from type (factory function)
    static CellType* create(Type type);

    /// Create cell type from string (factory function)
    static CellType* create(std::string type);

    /// Convert from string to cell type
    static Type string2type(std::string type);

    /// Convert from cell type to string
    static std::string type2string(Type type);

    /// Return type of cell
    Type cell_type() const { return _cell_type; }

    /// Return type of cell for facets
    Type facet_type() const { return _facet_type; }

    /// Return topological dimension of cell
    virtual uint dim() const = 0;

    /// Return number of entitites of given topological dimension
    virtual uint num_entities(uint dim) const = 0;

    /// Return number of vertices for entity of given topological dimension
    virtual uint num_vertices(uint dim) const = 0;

    /// Return orientation of the cell
    virtual uint orientation(const Cell& cell) const = 0;

    /// Create entities e of given topological dimension from vertices v
    virtual void create_entities(std::vector<std::vector<uint> >& e,
                                 uint dim, const uint* v) const = 0;

    /// Refine cell uniformly
    virtual void refine_cell(Cell& cell, MeshEditor& editor,
                             uint& current_cell) const = 0;

    /// Compute (generalized) volume of mesh entity
    virtual double volume(const MeshEntity& entity) const = 0;

    /// Compute diameter of mesh entity
    virtual double diameter(const MeshEntity& entity) const = 0;

    /// Compute component i of normal of given facet with respect to the cell
    virtual double normal(const Cell& cell, uint facet, uint i) const = 0;

    /// Compute of given facet with respect to the cell
    virtual Point normal(const Cell& cell, uint facet) const = 0;

    /// Compute the area/length of given facet with respect to the cell
    virtual double facet_area(const Cell& cell, uint facet) const = 0;

    // FIXME: The order() function should be reimplemented and use one common
    // FIXME: implementation for all cell types, just as we have for ordered()

    /// Order entities locally
    virtual void order(Cell& cell,
            const std::vector<uint>& local_to_global_vertex_indices) const = 0;

    /// Check if entities are ordered
    bool ordered(const Cell& cell,
                 const std::vector<uint>& local_to_global_vertex_indices) const;

    /// Return description of cell type
    virtual std::string description(bool plural) const = 0;

  protected:

    Type _cell_type;
    Type _facet_type;

    // Sort vertices based on global entity indices
    static void sort_entities(uint num_vertices,
                      uint* vertices,
                      const std::vector<uint>& local_to_global_vertex_indices);

  private:

    // Check if list of vertices is increasing
    static bool increasing(uint num_vertices, const uint* vertices,
                     const std::vector<uint>& local_to_global_vertex_indices);

    // Check that <entity e0 with vertices v0> <= <entity e1 with vertices v1>
    static bool increasing(uint n0, const uint* v0,
                       uint n1, const uint* v1,
                       uint num_vertices, const uint* vertices,
                       const std::vector<uint>& local_to_global_vertex_indices);

  };

}

#endif
