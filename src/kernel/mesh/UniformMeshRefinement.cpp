// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-08
// Last changed: 2006-06-12

#include <dolfin/dolfin_log.h>
#include <dolfin/NewMesh.h>
#include <dolfin/MeshData.h>
#include <dolfin/MeshTopology.h>
#include <dolfin/MeshGeometry.h>
#include <dolfin/MeshConnectivity.h>
#include <dolfin/MeshEditor.h>
#include <dolfin/NewVertex.h>
#include <dolfin/NewEdge.h>
#include <dolfin/NewCell.h>
#include <dolfin/UniformMeshRefinement.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void UniformMeshRefinement::refineInterval(NewMesh& mesh)
{
  dolfin_info("Refining interval mesh uniformly.");

}
//-----------------------------------------------------------------------------
void UniformMeshRefinement::refineTriangle(NewMesh& mesh)
{
  dolfin_info("Refining triangular mesh uniformly.");

  // Mesh needs to be of topological dimension 2
  dolfin_assert(mesh.dim() == 2);

  // Generate cell-edge connectivity if not generated
  mesh.init(2, 1);
  
  // Generate edge-vertex connectivity if not generated
  mesh.init(1, 0);
  
  // Create new mesh and open for editing
  NewMesh refined_mesh;
  MeshEditor editor;
  editor.edit(refined_mesh, 2, "triangle");

  // Get size of mesh
  uint num_vertices = mesh.numVertices();
  uint num_edges = mesh.numEdges();
  uint num_cells = mesh.numCells();

  // Specify number of vertices and cells
  editor.initVertices(num_vertices + num_edges);
  editor.initCells(4*num_cells);

  // Add old vertices
  uint vertex = 0;
  for (NewVertexIterator v(mesh); !v.end(); ++v)
    editor.addVertex(vertex++, v->x(), v->y());

  // Add new vertices
  for (NewEdgeIterator e(mesh); !e.end(); ++e)
  {
    NewPoint midpoint = e->midpoint();
    editor.addVertex(vertex++, midpoint.x(), midpoint.y());
  }    

  // Add cells
  uint cell = 0;
  for (NewCellIterator c(mesh); !c.end(); ++c)
  {
    // Get vertices and edges
    uint* vertices = c->connections(0);
    uint* edges = c->connections(1);
    dolfin_assert(vertices);
    dolfin_assert(edges);

    // Compute indices for the six new vertices
    uint v0 = vertices[0];
    uint v1 = vertices[1];
    uint v2 = vertices[2];
    uint e0 = num_vertices + edges[0];
    uint e1 = num_vertices + edges[1];
    uint e2 = num_vertices + edges[2];

    // Add the four new cells
    editor.addCell(cell++, v0, e2, e1);
    editor.addCell(cell++, v1, e0, e2);
    editor.addCell(cell++, v2, e1, e0);
    editor.addCell(cell++, e0, e1, e2);
  }

  editor.close();
  
  // Overwrite old mesh with refined mesh
  mesh = refined_mesh;
}
//-----------------------------------------------------------------------------
void UniformMeshRefinement::refineTetrahedron(NewMesh& mesh)
{
  dolfin_info("Refining tetrahedral mesh uniformly.");

}
//-----------------------------------------------------------------------------
