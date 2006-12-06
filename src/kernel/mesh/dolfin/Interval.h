// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-05
// Last changed: 2006-12-06

#ifndef __INTERVAL_H
#define __INTERVAL_H

#include <dolfin/CellType.h>

namespace dolfin
{

  /// This class implements functionality for interval meshes.

  class Interval : public CellType
  {
  public:

    /// Specify cell type and facet type
    Interval() : CellType(interval, point) {}

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

    /// Refine cell uniformly
    void refineCell(Cell& cell, MeshEditor& editor, uint& current_cell) const;

    /// Compute (generalized) volume (length) of interval
    real volume(const MeshEntity& interval) const;

    /// Compute diameter of interval
    real diameter(const MeshEntity& interval) const;

    /// Compute component i of normal of given facet with respect to the cell
    real normal(const Cell& cell, uint facet, uint i) const;

    /// Return description of cell type
    std::string description() const;

  };

}

#endif
