// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-06-05
// Last changed: 2008-06-20

#ifndef __INTERVAL_CELL_H
#define __INTERVAL_CELL_H

#include "CellType.h"

namespace dolfin
{

  /// This class implements functionality for interval meshes.

  class IntervalCell : public CellType
  {
  public:

    /// Specify cell type and facet type
    IntervalCell() : CellType(interval, point) {}

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

    /// Order entities locally (connectivity 1-0)
    void orderEntities(Cell& cell) const;

    /// Refine cell uniformly
    void refineCell(Cell& cell, MeshEditor& editor, uint& current_cell) const;

    /// Compute (generalized) volume (length) of interval
    real volume(const MeshEntity& interval) const;

    /// Compute diameter of interval
    real diameter(const MeshEntity& interval) const;

    /// Compute component i of normal of given facet with respect to the cell
    real normal(const Cell& cell, uint facet, uint i) const;

    /// Compute of given facet with respect to the cell
    Point normal(const Cell& cell, uint facet) const;

    /// Compute the area/length of given facet with respect to the cell
    real facetArea(const Cell& cell, uint facet) const;

    /// Check if point p intersects the cell
    bool intersects(const MeshEntity& entity, const Point& p) const;

    /// Check if points line connecting p1 and p2 cuts the cell
    bool intersects(const MeshEntity& entity, const Point& p1, const Point& p2) const;

    /// Return description of cell type
    std::string description() const;

  };

}

#endif
