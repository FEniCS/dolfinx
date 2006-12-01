// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Jansson 2006.
// Modified by Ola Skavhaug 2006.
//
// First added:  2006-06-21
// Last changed: 2006-12-01

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
void BoundaryComputation::computeBoundary(Mesh& mesh, BoundaryMesh& boundary)
{
  computeBoundaryCommon(mesh, boundary, 0, 0);
}
//-----------------------------------------------------------------------------
void BoundaryComputation::computeBoundary(Mesh& mesh, BoundaryMesh& boundary,
                                          MeshFunction<uint>& vertex_map,
                                          MeshFunction<uint>& cell_map)
{
  computeBoundaryCommon(mesh, boundary, &vertex_map, &cell_map);
}
//-----------------------------------------------------------------------------
void BoundaryComputation::computeBoundaryCommon(Mesh& mesh,
                                                BoundaryMesh& boundary,
                                                MeshFunction<uint>* vertex_map,
                                                MeshFunction<uint>* cell_map)
{
  // We iterate over all facets in the mesh and check if they are on
  // the boundary. A facet is on the boundary if it is connected to
  // exactly one cell.

  //dolfin_info("Computing boundary mesh.");

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
  uint num_boundary_cells = 0;
  for (FacetIterator f(mesh); !f.end(); ++f)
  {
    // Boundary facets are connected to exactly one cell
    if ( f->numEntities(mesh.topology().dim()) == 1 )
    {
      // Count boundary vertices and assign indices
      for (VertexIterator v(f); !v.end(); ++v)
      {
	const uint vertex_index = v->index();
	if ( boundary_vertices[vertex_index] == num_vertices )
	  boundary_vertices[vertex_index] = num_boundary_vertices++;
      }

      // Count boundary cells (facets of the mesh)
      num_boundary_cells++;
    }
  }
  
  // Specify number of vertices and cells
  editor.initVertices(num_boundary_vertices);
  editor.initCells(num_boundary_cells);

  // Initialize the mappings from boundary to mesh if requested
  if ( vertex_map )
    vertex_map->init(boundary, 0, num_boundary_vertices);
  if ( cell_map )
    cell_map->init(boundary, mesh.topology().dim() - 1, num_boundary_cells);
    
  // Create vertices
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    const uint vertex_index = boundary_vertices[v->index()];
    if ( vertex_index != mesh.numVertices() )
    {
      // Create mapping from boundary vertex to mesh vertex if requested
      if ( vertex_map )
        vertex_map->set(vertex_index, v->index());
      
      // Add vertex
      editor.addVertex(vertex_index, v->point());
    }
  }

  // Create cells (facets)
  Array<uint> cell(boundary.type().numVertices(boundary.topology().dim()));
  uint current_cell = 0;
  for (FacetIterator f(mesh); !f.end(); ++f)
  {
    // Boundary facets are connected to exactly one cell
    if ( f->numEntities(mesh.topology().dim()) == 1 )
    {
      // Compute new vertex numbers for cell
      uint* vertices = f->entities(0);
      for (uint i = 0; i < cell.size(); i++)
	cell[i] = boundary_vertices[vertices[i]];

      // Create mapping from boundary cell to mesh facet if requested
      if ( cell_map )
        cell_map->set(current_cell, f->index());

      // Add cell
      editor.addCell(current_cell++, cell);
    }
  }

  // Close mesh editor
  editor.close();

  cout << "Created boundary mesh: " << boundary << endl;
}
//-----------------------------------------------------------------------------
