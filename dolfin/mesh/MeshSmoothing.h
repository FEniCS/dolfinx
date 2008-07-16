// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-07-16
// Last changed: 2008-07-16

#ifndef __MESH_SMOOTHING_H
#define __MESH_SMOOTHING_H

namespace dolfin
{
  
  class Mesh;

  /// This class implements mesh smoothing. The coordinates of
  /// internal vertices are updated by local averaging.

  class MeshSmoothing
  {
  public:

    static void smooth(Mesh& mesh);

  };

}

#endif
