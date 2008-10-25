// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Jansson 2006.
// Modified by Ola Skavhaug 2006.
//
// First added:  2006-06-21
// Last changed: 2008-05-28

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Array.h>
#include "Mesh.h"
#include "Facet.h"
#include "Vertex.h"
#include "Cell.h"
#include "MeshEditor.h"
#include "MeshTopology.h"
#include "MeshGeometry.h"
#include "MeshData.h"
#include "BoundaryMesh.h"
#include "BoundaryComputation.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void BoundaryComputation::computeBoundary(const Mesh& mesh, BoundaryMesh& boundary)
{
  // We iterate over all facets in the mesh and check if they are on
  // the boundary. A facet is on the boundary if it is connected to
  // exactly one cell.

  message(1, "Computing boundary mesh.");

  // Open boundary mesh for editing
  const uint D = mesh.topology().dim();
  MeshEditor editor;
  editor.open(boundary, mesh.type().facetType(),
	      D - 1, mesh.geometry().dim());

  // Generate facet - cell connectivity if not generated
  mesh.init(D - 1, D);

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
    if (f->numEntities(D) == 1)
    {
      // Count boundary vertices and assign indices
      for (VertexIterator v(*f); !v.end(); ++v)
      {
	const uint vertex_index = v->index();
	if (boundary_vertices[vertex_index] == num_vertices)
	  boundary_vertices[vertex_index] = num_boundary_vertices++;
      }

      // Count boundary cells (facets of the mesh)
      num_boundary_cells++;
    }
  }
  
  // Specify number of vertices and cells
  editor.initVertices(num_boundary_vertices);
  editor.initCells(num_boundary_cells);

  // Initialize mapping from vertices in boundary to vertices in mesh
  MeshFunction<uint>* vertex_map = 0;
  if (num_boundary_vertices > 0)
  {
    vertex_map = boundary.data().createMeshFunction("vertex map");
    dolfin_assert(vertex_map);
    vertex_map->init(boundary, 0, num_boundary_vertices);
  }
  
  // Initialize mapping from cells in boundary to facets in mesh
  MeshFunction<uint>* cell_map = 0;
  if (num_boundary_cells > 0)
  {
    cell_map = boundary.data().createMeshFunction("cell map");
    dolfin_assert(cell_map);
    cell_map->init(boundary, D - 1, num_boundary_cells);
  }

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
    if (f->numEntities(D) == 1)
    {
      // Compute new vertex numbers for cell
      const uint* vertices = f->entities(0);
      for (uint i = 0; i < cell.size(); i++)
        cell[i] = boundary_vertices[vertices[i]];

      // Reorder vertices so facet is right-oriented w.r.t. facet normal
      reorder(cell, *f);

      // Create mapping from boundary cell to mesh facet if requested
      if (cell_map)
        cell_map->set(current_cell, f->index());

      // Add cell
      editor.addCell(current_cell++, cell);
    }
  }

  // Close mesh editor
  editor.close();
}
//-----------------------------------------------------------------------------
void BoundaryComputation::reorder(Array<uint>& vertices, Facet& facet)
{
  // Get mesh
  Mesh& mesh = facet.mesh();

  // Get the vertex opposite to the facet (the one we remove)
  uint vertex = 0;
  const Cell cell(mesh, facet.entities(mesh.topology().dim())[0]);
  for (uint i = 0; i < cell.numEntities(0); i++)
  {
    bool not_in_facet = true;
    vertex = cell.entities(0)[i];
    for (uint j = 0; j < facet.numEntities(0); j++)
    {
      if (vertex == facet.entities(0)[j])
      {
        not_in_facet = false;
        break;
      }
    }
    if (not_in_facet)
      break;
  }
  const Point p = mesh.geometry().point(vertex);

  // Check orientation
  switch (mesh.type().cellType())
  {
  case CellType::interval:
    // Do nothing
    break;
  case CellType::triangle:
    {
      dolfin_assert(facet.numEntities(0) == 2);
      
      Point p0 = mesh.geometry().point(facet.entities(0)[0]);
      Point p1 = mesh.geometry().point(facet.entities(0)[1]);
      Point v = p1 - p0;
      Point n(v.y(), -v.x());

      if (n.dot(p0 - p) < 0.0)
      {
        const uint tmp = vertices[0];
        vertices[0] = vertices[1];
        vertices[1] = tmp;
      }
    }
    break;
  case CellType::tetrahedron:
    {
      dolfin_assert(facet.numEntities(0) == 3);
    
      Point p0 = mesh.geometry().point(facet.entities(0)[0]);
      Point p1 = mesh.geometry().point(facet.entities(0)[1]);
      Point p2 = mesh.geometry().point(facet.entities(0)[2]);
      Point v1 = p1 - p0;
      Point v2 = p2 - p0;
      Point n = v1.cross(v2);

      if (n.dot(p0 - p) < 0.0)
      {
        const uint tmp = vertices[0];
        vertices[0] = vertices[1];
        vertices[1] = tmp;
      }
    }
    break;
  default:
    error("Unknown cell type, down know how to reorder.");
  }
}
//-----------------------------------------------------------------------------
