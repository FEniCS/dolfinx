// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-21
// Last changed: 2006-06-22

#include <dolfin/dolfin_log.h>
#include <dolfin/Array.h>
#include <dolfin/NewMesh.h>
#include <dolfin/NewFacet.h>
#include <dolfin/NewVertex.h>
#include <dolfin/MeshEditor.h>
#include <dolfin/MeshTopology.h>
#include <dolfin/MeshGeometry.h>
#include <dolfin/BoundaryMesh.h>
#include <dolfin/BoundaryComputation.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void BoundaryComputation::computeBoundary(NewMesh& mesh,
					  BoundaryMesh& boundary)
{
  // We iterate over all facets in the mesh and check if they are on
  // the boundary. A facet is on the boundary if it is connected to
  // exactly one cell.

  dolfin_info("Computing boundary mesh.");

  // Open boundary mesh for editing
  MeshEditor editor;
  editor.open(boundary, mesh.type().facetType(),
	      mesh.topology().dim() - 1, mesh.geometry().dim());

  // Generate facet - cell connectivity if not generated
  mesh.init(mesh.dim() - 1, mesh.dim());

  // Temporary array for assignment of indices to vertices on the boundary
  const uint num_vertices = mesh.numVertices();
  Array<uint> boundary_vertices(num_vertices);
  boundary_vertices = num_vertices;

  // Count boundary vertices and facets, and assign vertex indices
  uint num_boundary_vertices = 0;
  uint num_boundary_facets = 0;
  for (NewFacetIterator f(mesh); !f.end(); ++f)
  {
    // Boundary facets are connected to exactly one cell
    if ( f->numConnections(mesh.dim()) == 1 )
    {
      // Count boundary vertices and assign indices
      for (NewVertexIterator v(f); !v.end(); ++v)
      {
	const uint vertex_index = v->index();
	if ( boundary_vertices[vertex_index] == num_vertices )
	  boundary_vertices[vertex_index] = num_boundary_vertices++;
      }

      // Count boundary facets
      num_boundary_facets++;
    }
  }
  
  // Specify number of vertices and cells
  editor.initVertices(num_boundary_vertices);
  editor.initCells(num_boundary_facets);
  
  // Create vertices
  for (NewVertexIterator v(mesh); !v.end(); ++v)
  {
    const uint vertex_index = boundary_vertices[v->index()];
    if ( vertex_index != mesh.numVertices() )
      editor.addVertex(vertex_index, v->point());
  }

  // Create cells (facets)
  Array<uint> cell(boundary.type().numVertices(boundary.dim()));
  uint current_cell = 0;
  for (NewFacetIterator f(mesh); !f.end(); ++f)
  {
    // Boundary facets are connected to exactly one cell
    if ( f->numConnections(mesh.dim()) == 1 )
    {
      // Compute new vertex numbers for cell
      uint* vertices = f->connections(0);
      for (uint i = 0; i < cell.size(); i++)
	cell[i] = boundary_vertices[vertices[i]];

      // Add cell
      editor.addCell(current_cell++, cell);
    }
  }
  
  // Close mesh editor
  editor.close();

  cout << "Created boundary: " << boundary << endl;
}
//-----------------------------------------------------------------------------
