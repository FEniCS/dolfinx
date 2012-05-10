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
// First added:  2012-05-10
// Last changed: 2012-05-10

#ifndef __CSG_CGAL_MESH_GENERATOR3D_H
#define __CSG_CGAL_MESH_GENERATOR3D_H


namespace dolfin
{

  // Forward declarations
  class Mesh;
  class CSGGeometry;

  /// Mesh generator for Constructive Solid Geometry (CSG)
  /// utilizing CGALs boolean operation on Nef_polyhedrons.

  class CSGCGALMeshGenerator3D
  {
  public :
    CSGCGALMeshGenerator3D(const CSGGeometry& geometry);
    ~CSGCGALMeshGenerator3D();
    void generate(Mesh& mesh);

    //TODO: Add meshing parameters
  private:
    const CSGGeometry& geometry;
  };

}

#endif
