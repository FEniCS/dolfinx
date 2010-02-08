// Copyright (C) 2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-02-08
// Last changed: 2010-02-08

#ifndef __FACET_CELL_H
#define __FACET_CELL_H

#include "Cell.h"

namespace dolfin
{

  /// This class represents a cell in a mesh incident to a facet on
  /// the boundary. It is useful in cases where one needs to iterate
  /// over a boundary mesh and access the corresponding cells in the
  /// original mesh.

  class FacetCell : public Cell
  {
  public:

    /// Create cell on mesh corresponding to given facet (cell) on boundary
    FacetCell(const Mesh& mesh, const Cell& facet);

    /// Destructor
    ~FacetCell();

    /// Return local index of facet with respect to the cell
    uint facet_index() const;

  private:

    // Facet index
    uint _facet_index;

  };

}

#endif
