// Copyright (C) 2010-2011 Garth N. Wells
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
// Modified by Anders Logg, 2010.
//
// First added:  2010-11-15
// Last changed: 2011-01-29

#ifndef __MESH_COLORING_H
#define __MESH_COLORING_H

#include <string>
#include <vector>
#include <dolfin/common/types.h>

namespace dolfin
{

   class Mesh;

  /// This class computes colorings for a local mesh. It supports
  /// vertex, edge, and facet-based colorings.

  class MeshColoring
  {
  public:

    /// Color the cells of a mesh for given coloring type, which can
    /// be one of "vertex", "edge" or "facet".
    static const std::vector<uint>& color_cells(Mesh& mesh,
                                                 std::string coloring_type);

    /// Color the cells of a mesh for given coloring type specified by
    /// topological dimension, which can be one of 0, 1 or D - 1.
    static const std::vector<uint>& color(Mesh& mesh,
                                          std::vector<uint> coloring_type);

    /// Compute cell colors for given coloring type specified by
    /// topological dimension, which can be one of 0, 1 or D - 1.
    static uint compute_colors(const Mesh& mesh, std::vector<uint>& colors,
                               const std::vector<uint> coloring_type);

    /// Convert coloring type to topological dimension
    static uint type_to_dim(std::string coloring_type, const Mesh& mesh);

  };

}

#endif
