// Copyright (C) 2007-2007 Kristian B. Oelgaard.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
//
// First added:  2007-12-12
// Last changed: 2008-08-29

#ifndef __POINT_CELL_H
#define __POINT_CELL_H

#include "CellType.h"

namespace dolfin
{

  /// This class implements functionality for triangular meshes.

  class PointCell : public CellType
  {
  public:

    /// Specify cell type and facet type
    PointCell() : CellType(point, point) {}

    /// Return topological dimension of cell
    uint dim() const;

    /// Return number of entitites of given topological dimension
    uint numEntities(uint dim) const;

    /// Return number of vertices for entity of given topological dimension
    uint numVertices(uint dim) const;

    /// Return orientation of the cell
    uint orientation(const Cell& cell) const;

    /// Create entities e of given topological dimension from vertices v
    void createEntities(uint** e, uint dim, const uint* v) const;

    /// Order entities locally (connectivity 1-0, 2-0, 2-1)
    void orderEntities(Cell& cell) const;
    
    /// Refine cell uniformly
    void refineCell(Cell& cell, MeshEditor& editor, uint& current_cell) const;

    /// Compute (generalized) volume (area) of triangle
    real volume(const MeshEntity& triangle) const;

    /// Compute diameter of triangle
    real diameter(const MeshEntity& triangle) const;

    /// Compute component i of normal of given facet with respect to the cell
    real normal(const Cell& cell, uint facet, uint i) const;

    /// Compute of given facet with respect to the cell
    Point normal(const Cell& cell, uint facet) const;

    /// Compute the area/length of given facet with respect to the cell
    real facetArea(const Cell& cell, uint facet) const;

    /// Check for intersection with point
    bool intersects(const MeshEntity& entity, const Point& p) const;

    /// Check for intersection with line defined by points
    bool intersects(const MeshEntity& entity, const Point& p0, const Point& p1) const;
    
    /// Return description of cell type
    std::string description() const;

  private:

    // Find local index of edge i according to ordering convention
    uint findEdge(uint i, const Cell& cell) const;

  };

}

#endif
