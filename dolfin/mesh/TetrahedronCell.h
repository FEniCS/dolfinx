// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Hoffman, 2006.
// Modified by Garth N. Wells, 2006.
// Modified by Kristoffer Selim, 2008.
//
// First added:  2006-06-05
// Last changed: 2010-01-15

#ifndef __TETRAHEDRON_CELL_H
#define __TETRAHEDRON_CELL_H

#include "CellType.h"

namespace dolfin
{

  class Cell;

  /// This class implements functionality for tetrahedral meshes.

  class TetrahedronCell : public CellType
  {
  public:

    /// Specify cell type and facet type
    TetrahedronCell() : CellType(tetrahedron, triangle) {}

    /// Return topological dimension of cell
    uint dim() const;

    /// Return number of entitites of given topological dimension
    uint num_entities(uint dim) const;

    /// Return number of vertices for entity of given topological dimension
    uint num_vertices(uint dim) const;

    /// Return orientation of the cell
    uint orientation(const Cell& cell) const;

    /// Create entities e of given topological dimension from vertices v
    void create_entities(uint** e, uint dim, const uint* v) const;

    /// Regular refinement of cell
    void refine_cell(Cell& cell, MeshEditor& editor, uint& current_cell) const;

    /// Irregular refinement of cell
    void refine_cellIrregular(Cell& cell, MeshEditor& editor, uint& current_cell,
			     uint refinement_rule, uint* marked_edges) const;

    /// Compute volume of tetrahedron
    double volume(const MeshEntity& tetrahedron) const;

    /// Compute diameter of tetrahedron
    double diameter(const MeshEntity& tetrahedron) const;

    /// Compute component i of normal of given facet with respect to the cell
    double normal(const Cell& cell, uint facet, uint i) const;

    /// Compute normal of given facet with respect to the cell
    Point normal(const Cell& cell, uint facet) const;

    /// Compute the area/length of given facet with respect to the cell
    double facet_area(const Cell& cell, uint facet) const;

    /// Order entities locally
    void order(Cell& cell, const MeshFunction<uint>* global_vertex_indices) const;

    /// Return description of cell type
    std::string description(bool plural) const;

  private:

    // Find local index of edge i according to ordering convention
    uint find_edge(uint i, const Cell& cell) const;

  };

}

#endif
