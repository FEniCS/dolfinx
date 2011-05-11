// Copyright (C) 2006-2009 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Kristian B. Oelgaard, 2007.
// Modified by Martin Sandve Alnes, 2008.
// Modified by Garth N. Wells, 2008.
//
// First added:  2006-02-09
// Last changed: 2009-10-05

#ifndef __SPECIAL_FUNCTIONS_H
#define __SPECIAL_FUNCTIONS_H

#include "dolfin/common/Array.h"
#include "Expression.h"

namespace dolfin
{

  class Mesh;

  /// This Function represents the mesh coordinates on a given mesh.
  class MeshCoordinates : public Expression
  {
  public:

    /// Constructor
    MeshCoordinates(const Mesh& mesh);

    /// Evaluate function
    void eval(Array<double>& values, const Array<double>& x, const ufc::cell& cell) const;

  private:

    // The mesh
    const Mesh& mesh;

  };

  /*
  /// This Function represents the local cell size on a given mesh.
  class CellSize : public Expression
  {
  public:

    /// Constructor
    CellSize(const Mesh& mesh);

    /// Evaluate function
    void eval(Array<double>& values, const Data& data) const;

  private:

    // The mesh
    const Mesh& mesh;

  };
*/


  /// This function represents the area/length of a cell facet on a given mesh.
  class FacetArea : public Expression
  {
  public:

    /// Constructor
    FacetArea(const Mesh& mesh);

    /// Evaluate function
    void eval(Array<double>& values, const Array<double>& x,
              const ufc::cell& cell) const;

  private:

    // The mesh
    const Mesh& mesh;

  };

}

#endif
