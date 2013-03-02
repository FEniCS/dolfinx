// Copyright (C) 2012 Benjamin Kehlet
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
// Modified by Johannes Ring, 2012
//
// First added:  2012-05-10
// Last changed: 2012-05-14

#ifndef __CSG_CGAL_MESH_GENERATOR2D_H
#define __CSG_CGAL_MESH_GENERATOR2D_H

#include <dolfin/common/Variable.h>

namespace dolfin
{

  // Forward declarations
  class Mesh;
  class CSGGeometry;

  /// Mesh generator for Constructive Solid Geometry (CSG)
  /// utilizing CGALs boolean operation on Nef_polyhedrons.

  class CSGCGALMeshGenerator2D : public Variable
  {
  public :

    CSGCGALMeshGenerator2D(const CSGGeometry& geometry);

    ~CSGCGALMeshGenerator2D();

    void generate(Mesh& mesh);

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("csg_cgal_meshgenerator");
      p.add("triangle_shape_bound", 0.125);
      p.add("cell_size", 0.25);

      return p;
    }

  private:

    #ifdef HAS_CGAL
    const CSGGeometry& _geometry;
    #endif

  };

}

#endif
