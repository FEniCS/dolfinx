// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __MESH_INIT_H
#define __MESH_INIT_H

namespace dolfin{

  class Mesh;

  /// MeshInit implements the algorithm for computing the neighbour
  /// information (connections) in a mesh.
  ///
  /// The trick is to compute the connections in the correct order, as
  /// indicated in MeshData.h, to obtain an O(n) algorithm.
  
  class MeshInit {
  public:
    
    static void init     (Mesh& mesh);
    static void renumber (Mesh& mesh);
    
  private:
    
    static void clear            (Mesh& mesh);
    
    static void initConnectivity (Mesh& mesh);
    static void initEdges        (Mesh& mesh);
    static void initFaces        (Mesh& mesh);
    
    static void initNodeCell     (Mesh& mesh);
    static void initCellCell     (Mesh& mesh);
    static void initNodeEdge     (Mesh& mesh);
    static void initNodeNode     (Mesh& mesh);
    
    static void initEdgeCell     (Mesh& mesh);
    static void initFaceCell     (Mesh& mesh);

    static void initEdgeBoundaryids(Mesh& mesh);
    static void initFaceBoundaryids(Mesh& mesh);

  };

}

#endif
