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
// First added:  2010-10-19
// Last changed: 2010-10-19

#ifndef __UNIT_TETRAHEDRON_H
#define __UNIT_TETRAHEDRON_H

#include "Mesh.h"

namespace dolfin
{

  /// A mesh consisting of a single tetrahedron with vertices at
  ///
  ///   (0, 0, 0)
  ///   (1, 0, 0)
  ///   (0, 1, 0)
  ///   (0, 0, 1)
  ///
  /// This class is useful for testing.

  class UnitTetrahedron : public Mesh
  {
  public:

    /// Create mesh of unit tetrahedron
    UnitTetrahedron();

  };

}

#endif
