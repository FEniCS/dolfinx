// Copyright (C) 2008 Nuno Lopes
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
// First added:  2008-07-15
// Last changed: 2008-07-15

#ifndef __UNIT_SPHERE_H
#define __UNIT_SPHERE_H

#include "Mesh.h"

namespace dolfin
{

  /// Triangular mesh of the 3D unit sphere.
  ///
  /// Given the number of cells (nx, ny, nz) in each direction,
  /// the total number of tetrahedra will be 6*nx*ny*nz and the
  /// total number of vertices will be (nx + 1)*(ny + 1)*(nz + 1).


  class UnitSphere : public Mesh
  {
  public:

    UnitSphere(uint nx);

  private:

    double transformx(double x,double y,double z);
    double transformy(double x,double y,double z);
    double transformz(double x,double y,double z);
    double max(double x,double y,double z);

  };

}

#endif
