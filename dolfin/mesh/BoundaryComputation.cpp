// Copyright (C) 2006-2008 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Johan Jansson 2006.
// Modified by Ola Skavhaug 2006.
// Modified by Niclas Jansson 2009.
//
// First added:  2006-06-21
// Last changed: 2011-03-17

#include <dolfin/common/timing.h>

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
#include "ParallelData.h"
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

  log(TRACE, "Computing boundary mesh.");

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
  const MeshFunction<bool>& exterior = mesh.parallel_data().exterior_facet();

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
      if (exterior.empty())
        boundary_facet[*f] = true;
      else
      {
        bool exterior_facet = exterior[*f];
        if (exterior_facet && !interior_boundary)
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
  MeshFunction<unsigned int>& vertex_map = boundary.vertex_map();
  if (num_boundary_vertices > 0)
    vertex_map.init(boundary, 0, num_boundary_vertices);

  // Initialize mapping from cells in boundary to facets in mesh
  MeshFunction<unsigned int>& cell_map = boundary.cell_map();
  if (num_boundary_cells > 0)
    cell_map.init(boundary, D - 1, num_boundary_cells);

  // Create vertices
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    const uint vertex_index = boundary_vertices[v->index()];
    if (vertex_index != mesh.num_vertices())
    {
      // Create mapping from boundary vertex to mesh vertex if requested
      if (!vertex_map.empty())
        vertex_map[vertex_index] = v->index();

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
      if (!cell_map.empty())
        cell_map[current_cell] = f->index();

      // Add cell
      editor.add_cell(current_cell++, cell);
    }
  }

  // Close mesh editor. Note the argument order=false to prevent
  // ordering from destroying the orientation of facets accomplished
  // by calling reorder() below.
  editor.close(false);
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
      dolfin_assert(facet.num_entities(0) == 2);

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
      dolfin_assert(facet.num_entities(0) == 3);

      const Point p0 = mesh.geometry().point(facet.entities(0)[0]);
      const Point p1 = mesh.geometry().point(facet.entities(0)[1]);
      const Point p2 = mesh.geometry().point(facet.entities(0)[2]);
      const Point v1 = p1 - p0;
      const Point v2 = p2 - p0;
      const Point n  = v1.cross(v2);

      if (n.dot(p0 - p) < 0.0)
      {
        const uint tmp = vertices[0];
        vertices[0] = vertices[1];
        vertices[1] = tmp;
      }
    }
    break;
  default:
    {
      dolfin_error("BoundaryComputation.cpp",
                   "reorder cell for extraction of mesh boundary",
                   "Unknown cell type (%d)",
                   mesh.type().cell_type());
    }
  }
}
//-----------------------------------------------------------------------------
