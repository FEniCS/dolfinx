// Copyright (C) 2013 Garth N. Wells
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
// First added:  2013-05-05
// Last changed:

#ifndef __DOLFIN_SURFACEMESHGENERATOR_H
#define __DOLFIN_SURFACEMESHGENERATOR_H

#ifdef HAS_CGAL

namespace dolfin
{

  // Forward declarations
  class ImplicitSurface;
  class Mesh;

  /// This class generated DOLFIN meshes of closed in 3D surfaces. It
  /// uses CGAL for the mesh generation.

  class SurfaceMeshGenerator
  {
  public:

    /// Create mesh from an ImplicitSurface
    static void generate(Mesh& mesh, const ImplicitSurface& surface,
                         double min_angle, double max_radius,
                         double max_distance, std::size_t num_initial_points);

  };

}

#endif
#endif
