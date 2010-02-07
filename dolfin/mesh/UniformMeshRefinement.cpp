// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2010
//
// First added:  2006-06-08
// Last changed: 2010-02-07

#include <dolfin/math/dolfin_math.h>
#include <dolfin/log/dolfin_log.h>
#include "Mesh.h"
#include "MeshTopology.h"
#include "MeshGeometry.h"
#include "MeshConnectivity.h"
#include "MeshEditor.h"
#include "Vertex.h"
#include "Edge.h"
#include "Cell.h"
#include "UniformMeshRefinement.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::Mesh UniformMeshRefinement::refine(const Mesh& mesh)
{
  // Only know how to refine simplicial meshes
  return refine_simplex(mesh);
}
//-----------------------------------------------------------------------------
dolfin::Mesh UniformMeshRefinement::refine_simplex(const Mesh& mesh)
{
  info(1, "Refining simplicial mesh uniformly.");

  // Generate cell - edge connectivity if not generated
  mesh.init(mesh.topology().dim(), 1);

  // Generate edge - vertex connectivity if not generated
  mesh.init(1, 0);

  // Get cell type
  const CellType& cell_type = mesh.type();

  // Create new mesh and open for editing
  Mesh refined_mesh;
  MeshEditor editor;
  editor.open(refined_mesh, cell_type.cell_type(),
	            mesh.topology().dim(), mesh.geometry().dim());

  // Get size of mesh
  const uint num_vertices = mesh.size(0);
  const uint num_edges = mesh.size(1);
  const uint num_cells = mesh.size(mesh.topology().dim());

  // Specify number of vertices and cells
  editor.init_vertices(num_vertices + num_edges);
  editor.init_cells(ipow(2, mesh.topology().dim())*num_cells);

  // Add old vertices
  uint vertex = 0;
  for (VertexIterator v(mesh); !v.end(); ++v)
    editor.add_vertex(vertex++, v->point());

  // Add new vertices
  for (EdgeIterator e(mesh); !e.end(); ++e)
    editor.add_vertex(vertex++, e->midpoint());

  // Add cells
  uint current_cell = 0;
  for (CellIterator c(mesh); !c.end(); ++c)
    cell_type.refine_cell(*c, editor, current_cell);

  // Close editor
  editor.close();

  return refined_mesh;
}
//-----------------------------------------------------------------------------
