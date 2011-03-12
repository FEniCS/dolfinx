// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Jansson 2006.
// Modified by Ola Skavhaug 2006.
// Modified by Niclas Jansson 2009.
//
// First added:  2006-06-21
// Last changed: 2010-03-02

#include <dolfin/log/dolfin_log.h>
#include "BoundaryMesh.h"
#include "Cell.h"
#include "Facet.h"
#include "Mesh.h"
#include "MeshData.h"
#include "MeshEditor.h"
#include "MeshEntity.h"
#include "MeshFunction.h"
#include "MeshGeometry.h"
#include "MeshTopology.h"
#include "Vertex.h"
#include "BoundaryComputation.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void BoundaryComputation::compute_exterior_boundary(const Mesh& mesh,
                                                    BoundaryMesh& boundary)
{
  compute_boundary_common(mesh, boundary, false);
}
//-----------------------------------------------------------------------------
void BoundaryComputation::compute_interior_boundary(const Mesh& mesh,
                                                    BoundaryMesh& boundary)
{
  compute_boundary_common(mesh, boundary, true);
}
//-----------------------------------------------------------------------------
void BoundaryComputation::compute_boundary_common(const Mesh& mesh,
                                                  BoundaryMesh& boundary,
                                                  bool interior_boundary)
{
  // We iterate over all facets in the mesh and check if they are on
  // the boundary. A facet is on the boundary if it is connected to
  // exactly one cell.

  info(TRACE, "Computing boundary mesh.");

  // Open boundary mesh for editing
  const uint D = mesh.topology().dim();
  MeshEditor editor;
  editor.open(boundary, mesh.type().facet_type(), D - 1, mesh.geometry().dim());

  // Generate facet - cell connectivity if not generated
  mesh.init(D - 1, D);

  // Temporary array for assignment of indices to vertices on the boundary
  const uint num_vertices = mesh.num_vertices();
  std::vector<uint> boundary_vertices(num_vertices);
  std::fill(boundary_vertices.begin(), boundary_vertices.end(), num_vertices);

  // Extract exterior (non shared) facets markers
  boost::shared_ptr<const MeshFunction<unsigned int> > exterior = mesh.data().mesh_function("exterior facets");

  // Determine boundary facet, count boundary vertices and facets,
  // and assign vertex indices
  uint num_boundary_vertices = 0;
  uint num_boundary_cells = 0;
  MeshFunction<bool> boundary_facet(mesh, D - 1, false);
  for (FacetIterator f(mesh); !f.end(); ++f)
  {
    // Boundary facets are connected to exactly one cell
    if (f->num_entities(D) == 1)
    {
      // Determine if we have a boundary facet
      if (!exterior)
        boundary_facet[*f] = true;
      else
      {
        bool exterior_facet = (*exterior)[*f];
        if ( exterior_facet && !interior_boundary )
          boundary_facet[*f] = true;
        else if ( !exterior_facet && interior_boundary )
          boundary_facet[*f] = true;
      }

      if (boundary_facet[*f])
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
  }

  // Specify number of vertices and cells
  editor.init_vertices(num_boundary_vertices);
  editor.init_cells(num_boundary_cells);

  // Initialize mapping from vertices in boundary to vertices in mesh
   boost::shared_ptr<MeshFunction<unsigned int> > vertex_map = boundary.data().create_mesh_function("vertex map");
  assert(vertex_map);
  if (num_boundary_vertices > 0)
    vertex_map->init(boundary, 0, num_boundary_vertices);

  // Initialize mapping from cells in boundary to facets in mesh
   boost::shared_ptr<MeshFunction<unsigned int> > cell_map = boundary.data().create_mesh_function("cell map");
  assert(cell_map);
  if (num_boundary_cells > 0)
    cell_map->init(boundary, D - 1, num_boundary_cells);

  // Create vertices
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    const uint vertex_index = boundary_vertices[v->index()];
    if (vertex_index != mesh.num_vertices())
    {
      // Create mapping from boundary vertex to mesh vertex if requested
      if (vertex_map->size() > 0)
        (*vertex_map)[vertex_index] = v->index();

      // Add vertex
      editor.add_vertex(vertex_index, v->point());
    }
  }

  // Create cells (facets)
  std::vector<uint> cell(boundary.type().num_vertices(boundary.topology().dim()));
  uint current_cell = 0;
  for (FacetIterator f(mesh); !f.end(); ++f)
  {
    if (boundary_facet[*f])
    {
      // Compute new vertex numbers for cell
      const uint* vertices = f->entities(0);
      for (uint i = 0; i < cell.size(); i++)
        cell[i] = boundary_vertices[vertices[i]];

      // Reorder vertices so facet is right-oriented w.r.t. facet normal
      reorder(cell, *f);

      // Create mapping from boundary cell to mesh facet if requested
      if (cell_map->size() > 0)
        (*cell_map)[current_cell] = f->index();

      // Add cell
      editor.add_cell(current_cell++, cell);
    }
  }

  // Close mesh editor
  editor.close();
}
//-----------------------------------------------------------------------------
void BoundaryComputation::reorder(std::vector<uint>& vertices,
                                  const Facet& facet)
{
  // Get mesh
  const Mesh& mesh = facet.mesh();

  // Get the vertex opposite to the facet (the one we remove)
  uint vertex = 0;
  const Cell cell(mesh, facet.entities(mesh.topology().dim())[0]);
  for (uint i = 0; i < cell.num_entities(0); i++)
  {
    bool not_in_facet = true;
    vertex = cell.entities(0)[i];
    for (uint j = 0; j < facet.num_entities(0); j++)
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
  switch (mesh.type().cell_type())
  {
  case CellType::interval:
    // Do nothing
    break;
  case CellType::triangle:
    {
      assert(facet.num_entities(0) == 2);

      const Point p0 = mesh.geometry().point(facet.entities(0)[0]);
      const Point p1 = mesh.geometry().point(facet.entities(0)[1]);
      const Point v = p1 - p0;
      const Point n(v.y(), -v.x());

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
      assert(facet.num_entities(0) == 3);

      const Point p0 = mesh.geometry().point(facet.entities(0)[0]);
      const Point p1 = mesh.geometry().point(facet.entities(0)[1]);
      const Point p2 = mesh.geometry().point(facet.entities(0)[2]);
      const Point v1 = p1 - p0;
      const Point v2 = p2 - p0;
      const Point n = v1.cross(v2);

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
