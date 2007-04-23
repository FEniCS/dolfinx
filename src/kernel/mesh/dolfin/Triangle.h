// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-06-05
// Last changed: 2007-01-30

#ifndef __TRIANGLE_H
#define __TRIANGLE_H

#include <dolfin/CellType.h>

namespace dolfin
{

  /// This class implements functionality for triangular meshes.

  class Triangle : public CellType
  {
  public:

    /// Specify cell type and facet type
    Triangle() : CellType(triangle, interval) {}

    /// Return topological dimension of cell
    uint dim() const;

    /// Return number of entitites of given topological dimension
    uint numEntities(uint dim) const;

    /// Return number of vertices for entity of given topological dimension
    uint numVertices(uint dim) const;

    /// Return alignment of given entity with respect to the cell
    uint alignment(const Cell& cell, uint dim, uint e) const;

    /// Create entities e of given topological dimension from vertices v
    void createEntities(uint** e, uint dim, const uint v[]) const;

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

    /// Check if point p intersects the cell
    bool intersects(const MeshEntity& entity, const Point& p) const;

    /// Return description of cell type
    std::string description() const;

  };

}

#endif
