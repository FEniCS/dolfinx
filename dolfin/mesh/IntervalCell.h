// Copyright (C) 2006-2008 Anders Logg
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
//
// First added:  2006-06-05
// Last changed: 2010-01-15

#ifndef __INTERVAL_CELL_H
#define __INTERVAL_CELL_H

#include <string>
#include <vector>
#include <dolfin/common/types.h>
#include "CellType.h"

namespace dolfin
{

  class Cell;
  class MeshEditor;
  class MeshEntity;
  template<typename T> class MeshFunction;

  /// This class implements functionality for interval meshes.

  class IntervalCell : public CellType
  {
  public:

    /// Specify cell type and facet type
    IntervalCell() : CellType(interval, point) {}

    /// Return topological dimension of cell
    uint dim() const;

    /// Return number of entitites of given topological dimension
    uint num_entities(uint dim) const;

    /// Return number of vertices for entity of given topological dimension
    uint num_vertices(uint dim) const;

    /// Return orientation of the cell
    uint orientation(const Cell& cell) const;

    /// Create entities e of given topological dimension from vertices v
    void create_entities(std::vector<std::vector<uint> >& e, uint dim, const uint* v) const;

    /// Refine cell uniformly
    void refine_cell(Cell& cell, MeshEditor& editor, uint& current_cell) const;

    /// Compute (generalized) volume (length) of interval
    double volume(const MeshEntity& interval) const;

    /// Compute diameter of interval
    double diameter(const MeshEntity& interval) const;

    /// Compute component i of normal of given facet with respect to the cell
    double normal(const Cell& cell, uint facet, uint i) const;

    /// Compute of given facet with respect to the cell
    Point normal(const Cell& cell, uint facet) const;

    /// Compute the area/length of given facet with respect to the cell
    double facet_area(const Cell& cell, uint facet) const;

    /// Order entities locally
    void order(Cell& cell, const MeshFunction<uint>* global_vertex_indices) const;

    /// Return description of cell type
    std::string description(bool plural) const;

  };

}

#endif
