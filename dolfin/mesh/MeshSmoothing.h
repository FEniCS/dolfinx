// Copyright (C) 2008-2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
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
    static void smooth(Mesh& mesh, uint num_iterations=1);

    /// Smooth boundary vertices of mesh by local averaging and
    /// (optionally) use harmonic smoothing on interior vertices
    static void smooth_boundary(Mesh& mesh,
                                uint num_iterations=1,
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
