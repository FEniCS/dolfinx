// Copyright (C) 2006 Johan Hoffman.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-11-01

#ifndef __LOCAL_MESH_REFINEMENT_H
#define __LOCAL_MESH_REFINEMENT_H

#include "MeshFunction.h"

namespace dolfin
{

  class Cell;
  class Edge;
  class Mesh;
  class MeshEditor;

  /// This class implements local mesh refinement for different mesh types.

  class LocalMeshRefinement
  {
  public:

    /// Refine simplicial mesh locally by edge bisection 
    static void refineMeshByEdgeBisection(Mesh& mesh, 
                                          MeshFunction<bool>& cell_marker,
                                          bool refine_boundary = true); 
    
    /// Iteratively refine mesh locally by the longest edge bisection 
    static void refineIterativelyByEdgeBisection(Mesh& mesh, 
                                                 MeshFunction<bool>& cell_marker);
 
    /// Recursively refine mesh locally by the longest edge bisection 
    /// Fast Rivara algorithm implementation with propagation MeshFunctions and
    ///   arrays for boundary indicators  
    static void refineRecursivelyByEdgeBisection(Mesh& mesh, 
                                                 MeshFunction<bool>& cell_marker);
 
  private: 

    /// Bisect edge of simplex cell
    static void bisectEdgeOfSimplexCell(const Cell& cell, Edge& edge, 
                                        uint new_vertex,  
                                        MeshEditor& editor, 
                                        uint& current_cell);
 
    /// Iteration of iterative algorithm                                        
    static bool iterationOfRefinement(Mesh& mesh, 
                                      MeshFunction<bool>& cell_marker,
                                      MeshFunction<uint>& bisected_edges);

    /// Transform MeshData
    static void transformMeshData(Mesh& newmesh, Mesh& oldmesh, 
                                  MeshFunction<uint>& cell_map,
		                  std::vector<int>& facet_map);                                    

  };

}

#endif
