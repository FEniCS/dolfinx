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
// Modified by Anders Logg 2012
//
// First added:  2008-07-15
// Last changed: 2012-03-06

#ifndef __UNIT_SPHERE_H
#define __UNIT_SPHERE_H

#include <dolfin/mesh/Mesh.h>

namespace dolfin
{

  /// Tetrahedral mesh of the unit sphere.

  class UnitSphere : public Mesh
  {
  public:

    /// WARNING:
    ///
    /// The UnitSphere class is broken and should not be used for computations.
    /// It generates meshes of very bad quality (very thin tetrahedra).
    ///
    /// Create a uniform finite element _Mesh_ over the unit sphere.
    ///
    /// *Arguments*
    ///     n (uint)
    ///         Resolution of the mesh.
    UnitSphere(uint n);

  private:

    double transformx(double x,double y,double z);

    double transformy(double x,double y,double z);

    double transformz(double x,double y,double z);

    double max(double x,double y,double z);

  };

}

#endif
