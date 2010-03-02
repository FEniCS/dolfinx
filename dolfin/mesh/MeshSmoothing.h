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

  /// This class implements various mesh smoothing algorithms.

  class MeshSmoothing
  {
  public:

    /// Smooth internal vertices of mesh by local averaging
    static void smooth(Mesh& mesh);

    /// Smooth boundary of mesh by local averaging
    static void smooth_boundary(Mesh& mesh);

  };

}

#endif
