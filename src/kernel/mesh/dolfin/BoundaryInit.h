// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __BOUNDARY_INIT_H
#define __BOUNDARY_INIT_H

namespace dolfin {

  class Mesh;

  /// BoundaryInit implements the algorithm for computing the boundary
  /// of a given mesh.

  class BoundaryInit {
  public:
    
    static void init(Mesh& mesh);
    
  private:

    static void clear(Mesh& mesh);

    static void initFaces(Mesh& mesh);
    static void initEdges(Mesh& mesh);
    static void initNodes(Mesh& mesh);

    static void initFacesTri(Mesh& mesh);
    static void initFacesTet(Mesh& mesh);
    static void initEdgesTri(Mesh& mesh);
    static void initEdgesTet(Mesh& mesh);

  };

}

#endif
