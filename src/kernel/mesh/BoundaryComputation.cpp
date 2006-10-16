// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-21
// Last changed: 2006-10-11

#include <dolfin/dolfin_log.h>
#include <dolfin/Array.h>
#include <dolfin/Mesh.h>
#include <dolfin/Facet.h>
#include <dolfin/Vertex.h>
#include <dolfin/Cell.h>
#include <dolfin/MeshEditor.h>
#include <dolfin/MeshTopology.h>
#include <dolfin/MeshGeometry.h>
#include <dolfin/BoundaryMesh.h>
#include <dolfin/BoundaryComputation.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void BoundaryComputation::computeBoundary(Mesh& mesh,
					  BoundaryMesh& boundary,
					  Array<uint>& icell)
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
  mesh.init(mesh.topology().dim() - 1, mesh.topology().dim());

  // Temporary array for assignment of indices to vertices on the boundary
  const uint num_vertices = mesh.numVertices();
  Array<uint> boundary_vertices(num_vertices);
  boundary_vertices = num_vertices;

  // Count boundary vertices and facets, and assign vertex indices
  uint num_boundary_vertices = 0;
  uint num_boundary_facets = 0;
  for (FacetIterator f(mesh); !f.end(); ++f)
  {
    // Boundary facets are connected to exactly one cell
    if ( f->numConnections(mesh.topology().dim()) == 1 )
    {
      // Count boundary vertices and assign indices
      for (VertexIterator v(f); !v.end(); ++v)
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
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    const uint vertex_index = boundary_vertices[v->index()];
    if ( vertex_index != mesh.numVertices() )
      editor.addVertex(vertex_index, v->point());
  }

  icell.resize(num_boundary_facets);

  // Create cells (facets)
  Array<uint> cell(boundary.type().numVertices(boundary.topology().dim()));
  uint current_cell = 0;
  for (FacetIterator f(mesh); !f.end(); ++f)
  {
    // Boundary facets are connected to exactly one cell
    if ( f->numConnections(mesh.topology().dim()) == 1 )
    {
      // Compute new vertex numbers for cell
      uint* vertices = f->connections(0);
      for (uint i = 0; i < cell.size(); i++)
	cell[i] = boundary_vertices[vertices[i]];

      // Add cell
      editor.addCell(current_cell++, cell);

      // Add cell to interior map
      icell[current_cell - 1] = f->connections(mesh.topology().dim())[0];
    }
  }

  // Close mesh editor
  editor.close();

//   editor.open(boundary, mesh.type().cellType(),
// 	      mesh.topology().topology().dim(), mesh.geometry().topology().dim());

//   // Create interior cells touching boundary
//   Array<bool> icells(mesh.numCells());
//   for(uint i = 0; i < icells.size(); i++)
//   {
//     icells[i] = false;
//   }

//   for(CellIterator c(boundary); !c.end(); ++c)
//   {
//     cout << "bcell: " << c->index() << endl;
//   }

//   editor.close();
  
  cout << "Created boundary: " << boundary << endl;
}
//-----------------------------------------------------------------------------
