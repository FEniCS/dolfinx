// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version
//
// Modified by Anders Logg, 2005.

#ifndef __MESH_REFINEMENT_H
#define __MESH_REFINEMENT_H

#include <dolfin/PList.h>

namespace dolfin
{

  class Mesh;
  class Cell;
  class MeshHierarchy;

  /// Algorithm for the refinement of a triangular or tetrahedral mesh.
  
  class MeshRefinement
  {
  public:

    /// Refine a given mesh hierarchy according to marks
    static void refine(MeshHierarchy& meshes);
    
  protected:
    
    //--- Algorithms working on the whole mesh hierarchy ---

    // Propagate markers for leaf elements
    static void propagateLeafMarks(MeshHierarchy& meshes);
    
    /// Update marks for edges
    static void updateEdgeMarks(MeshHierarchy& meshes);

    // The global algorithm
    static void globalRefinement(MeshHierarchy& meshes);

    // Check consistency of markers before refinement
    static void checkPreCondition(MeshHierarchy& meshes);

    // Check consistency of markers after refinement
    static void checkPostCondition(MeshHierarchy& meshes);

    // Check object numbering
    static void checkNumbering(MeshHierarchy& meshes);

    //--- Algorithms working on the mesh on a given level ---
    
    /// Update marks for edges
    static void updateEdgeMarks(Mesh& mesh);

    // Evaluate and adjust marks for a mesh
    static void evaluateMarks(Mesh& mesh);

    // Perform the green closer on a mesh
    static void closeMesh(Mesh& mesh);

    // Refine a mesh according to marks
    static void refineMesh(Mesh& mesh);

    // Unrefine a mesh according to marks
    static void unrefineMesh(Mesh& mesh, const MeshHierarchy& meshes);

    ///--- Algorithms working on a given cell ---
    
    // Close a cell
    static void closeCell(Cell& cell, PList<Cell*>& cells, PArray<bool>& closed);

    /// Check refinement rule for given cell
    static bool checkRule(Cell& cell, int no_marked_edges);
    
    // Refine cell according to refinement rule
    static void refine(Cell& cell, Mesh& mesh);
    
    ///--- A couple of special functions, placed here rather than in Cell ---

    /// Update marks for edges
    static void updateEdgeMarks(Cell& cell);
    
    /// Check if all children are marked for coarsening
    static bool childrenMarkedForCoarsening(Cell& cell);

    /// Check if at least one edge of a child is marked for refinement
    static bool edgeOfChildMarkedForRefinement(Cell& cell);

    /// Check if the cell has at least one edge marked by another cell (but not the cell itself)
    static bool edgeMarkedByOther(Cell& cell);

    /// Sort nodes, placing the node belonging to the most number of marked edges first
    static void sortNodes(const Cell& cell, PArray<Node*>& nodes);
    
    /// Count the number of marked edges within a cell
    static int noMarkedEdges(const Cell& cell);

    /// Mapping from global node number to local number within cell
    static int nodeNumber(const Node& node, const Cell& cell);

    /// Check if a given cell is a leaf element
    static bool leaf(Cell& cell);
    
    /// Check if cell is allowed to be refined
    static bool okToRefine(Cell& cell);

    /// Create a new node (if it doesn't exist) and set parent-child info
    static Node& createNode(Node& node, Mesh& mesh, const Cell& cell);
    
    /// Create a new node (if it doesn't exist)
    static Node& createNode(const Point& p, Mesh& mesh, const Cell& cell);
    
    /// Remove node 
    static void removeNode(Node& node, Mesh& mesh);

    /// Remove cell 
    static void removeCell(Cell& cell, Mesh& mesh);

    /// Create a new child to cell, that is a copy of cell 
    static Cell& createChildCopy(Cell& cell, Mesh& mesh);

  };

}

#endif
