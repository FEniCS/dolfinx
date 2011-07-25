// Copyright (C) 2005-2009 Anders Logg
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
// Modified by Nuno Lopes, 2008.
//
// First added:  2005-12-02
// Last changed: 2009-02-11

#ifndef __BOX_H
#define __BOX_H

#include "Mesh.h"

namespace dolfin
{

  /// Tetrahedral mesh of the 3D rectangular prism [x0, x1] x [y0, y1]
  /// x [z0, z1].  Given the number of cells (nx, ny, nz) in each
  /// direction, the total number of tetrahedra will be 6*nx*ny*nz and
  /// the total number of vertices will be (nx + 1)*(ny + 1)*(nz + 1).

  class Box : public Mesh
  {
  public:

    /// Define a uniform finite element _Mesh_ over the rectangular prism
    /// [x0, x1] x [y0, y1] x [z0, z1].
    ///
    /// *Arguments*
    ///     x0 (double)
    ///         :math:`x`-min.
    ///     x1 (double)
    ///         :math:`x`-max.
    ///     y0 (double)
    ///         :math:`y`-min.
    ///     y1 (double)
    ///         :math:`y`-max.
    ///     z0 (double)
    ///         :math:`z`-min.
    ///     z1 (double)
    ///         :math:`z`-max.
    ///     xn (double)
    ///         Number of cells in :math:`x`-direction.
    ///     yn (double)
    ///         Number of cells in :math:`y`-direction.
    ///     zn (double)
    ///         Number of cells in :math:`z`-direction.
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         // Mesh with 6 cells in each direction on the
    ///         // interval [-1,2] x [-1,2] x [-1,2].
    ///         Box mesh(-1, -1, -1, 2, 2, 2, 6, 6, 6);
    ///
    Box(double x0, double y0, double z0, double x1, double y1, double z1,
        uint nx, uint ny, uint nz);

  };

}

#endif
