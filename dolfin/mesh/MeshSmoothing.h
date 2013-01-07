// Copyright (C) 2008-2010 Anders Logg
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
// First added:  2008-07-16
// Last changed: 2010-03-02

#ifndef __MESH_SMOOTHING_H
#define __MESH_SMOOTHING_H

namespace dolfin
{

  class Mesh;
  class SubDomain;

  /// This class implements various mesh smoothing algorithms.

  class MeshSmoothing
  {
  public:

    /// Smooth internal vertices of mesh by local averaging
    static void smooth(Mesh& mesh, std::size_t num_iterations=1);

    /// Smooth boundary vertices of mesh by local averaging and
    /// (optionally) use harmonic smoothing on interior vertices
    static void smooth_boundary(Mesh& mesh,
                                std::size_t num_iterations=1,
                                bool harmonic_smoothing=true);

    /// Snap boundary vertices of mesh to match given sub domain and
    /// (optionally) use harmonic smoothing on interior vertices
    static void snap_boundary(Mesh& mesh,
                              const SubDomain& sub_domain,
                              bool harmonic_smoothing=true);

  private:

    // Move interior vertices
    static void move_interior_vertices(Mesh& mesh,
                                       BoundaryMesh& boundary,
                                       bool harmonic_smoothing);

  };

}

#endif
