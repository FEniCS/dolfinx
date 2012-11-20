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
// Modified by Garth N. Wells, 2011.
//
// First added:  2010-11-27
// Last changed: 2011-01-16

#ifndef __MESH_RENUMBERING_H
#define __MESH_RENUMBERING_H

#include <vector>
#include "dolfin/common/types.h"

namespace dolfin
{

  class Mesh;

  /// This class implements renumbering algorithms for meshes.

  class MeshRenumbering
  {
  public:

    /// Renumber mesh entities by coloring. This function is currently
    /// restricted to renumbering by cell coloring. The cells
    /// (cell-vertex connectivity) and the coordinates of the mesh are
    /// renumbered to improve the locality within each color. It is
    /// assumed that the mesh has already been colored and that only
    /// cell-vertex connectivity exists as part of the mesh.
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         Mesh to be renumbered.
    ///     coloring (_std::vector<std::size_t>_)
    ///         Mesh coloring type.
    /// *Returns*
    ///     _Mesh_
    static Mesh renumber_by_color(const Mesh& mesh,
                                  std::vector<std::size_t> coloring);

  private:

    static void compute_renumbering(const Mesh& mesh,
                                    const std::vector<std::size_t>& coloring,
                                    std::vector<double>& coordinates,
                                    std::vector<uint>& connectivity);


  };

}

#endif
