// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-05
// Last changed: 2006-06-22

#ifndef __NEW_TETRAHEDRON_H
#define __NEW_TETRAHEDRON_H

#include <dolfin/CellType.h>

namespace dolfin
{

  /// This class implements functionality for tetrahedral meshes.

  class NewTetrahedron : public CellType
  {
  public:

    /// Return type of cell
    inline Type cellType() const { return tetrahedron; }

    /// Return type of cell for facets
    inline Type facetType() const { return triangle; }

    /// Return number of entitites of given topological dimension
    uint numEntities(uint dim) const;

    /// Return number of vertices for entity of given topological dimension
    uint numVertices(uint dim) const;

    /// Create entities e of given topological dimension from vertices v
    void createEntities(uint** e, uint dim, const uint v[]) const;

    /// Refine cell uniformly
    void refineCell(NewCell& cell, MeshEditor& editor, uint& current_cell) const;

    /// Return description of cell type
    std::string description() const;

  private:

  };

}

#endif
