// Copyright (C) 2010 Anders Logg
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
// First added:  2010-02-08
// Last changed: 2010-02-08

#ifndef __FACET_CELL_H
#define __FACET_CELL_H

#include "Cell.h"

namespace dolfin
{

  class BoundaryMesh;

  /// This class represents a cell in a mesh incident to a facet on
  /// the boundary. It is useful in cases where one needs to iterate
  /// over a boundary mesh and access the corresponding cells in the
  /// original mesh.

  class FacetCell : public Cell
  {
  public:

    /// Create cell on mesh corresponding to given facet (cell) on boundary
    FacetCell(const BoundaryMesh& mesh, const Cell& facet);

    /// Destructor
    ~FacetCell();

    /// Return local index of facet with respect to the cell
    std::size_t facet_index() const;

  private:

    // Facet index
    std::size_t _facet_index;

  };

}

#endif
