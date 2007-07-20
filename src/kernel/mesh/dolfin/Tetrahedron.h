// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Hoffman 2006.
// Modified by Garth N. Wells 2006.
//
// First added:  2006-06-05
// Last changed: 2007-07-20

#ifndef __TETRAHEDRON_H
#define __TETRAHEDRON_H

#include <dolfin/CellType.h>

namespace dolfin
{

  class Cell;

  /// This class implements functionality for tetrahedral meshes.

  class Tetrahedron : public CellType
  {
  public:

    /// Specify cell type and facet type
    Tetrahedron() : CellType(tetrahedron, triangle) {}

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

    /// Order entities locally (connectivity 1-0, 2-0, 2-1, 3-0, 3-1, 3-2)
    void orderEntities(Cell& cell) const;

    /// Regular refinement of cell 
    void refineCell(Cell& cell, MeshEditor& editor, uint& current_cell) const;

    /// Irregular refinement of cell 
    void refineCellIrregular(Cell& cell, MeshEditor& editor, uint& current_cell, 
			     uint refinement_rule, uint* marked_edges) const;

    /// Compute volume of tetrahedron
    real volume(const MeshEntity& tetrahedron) const;

    /// Compute diameter of tetrahedron
    real diameter(const MeshEntity& tetrahedron) const;

    /// Compute component i of normal of given facet with respect to the cell
    real normal(const Cell& cell, uint facet, uint i) const;

    /// Check if point p intersects the cell
    bool intersects(const MeshEntity& entity, const Point& p) const;

    /// Return description of cell type
    std::string description() const;

  private:
    
    // Find local index of edge i according to ordering convention
    uint findEdge(uint i, const Cell& cell) const;

  };

}

#endif
