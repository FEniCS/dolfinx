// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-02-11
// Last changed: 2009-02-11

#include <set>
#include <vector>

#include "Cell.h"
#include "Vertex.h"
#include "MeshEditor.h"
#include "SubDomain.h"
#include "SubMesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
SubMesh::SubMesh(const Mesh& mesh, const SubDomain& subdomain)
{
  // Open mesh for editing
  MeshEditor editor;
  editor.open(*this, mesh.type().cellType(),
              mesh.topology().dim(), mesh.geometry().dim());

  // Extract cells
  std::set<uint> cells;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    if (subdomain.inside(cell->midpoint().coordinates(), false))
      cells.insert(cell->index());
  }

  // Add cells
  editor.initCells(cells.size());
  uint current_cell = 0;
  std::set<uint> vertices;
  for (std::set<uint>::iterator it = cells.begin(); it != cells.end(); ++it)
  {
    std::vector<uint> cell_vertices;
    Cell cell(mesh, *it);
    for (VertexIterator vertex(cell); !vertex.end(); ++vertex)
    {
      vertices.insert(vertex->index());
      cell_vertices.push_back(vertex->index());
    }
    editor.addCell(current_cell++, cell_vertices);
  }

  // Add vertices
  editor.initVertices(vertices.size());
  uint current_vertex = 0;
  for (std::set<uint>::iterator it = vertices.begin(); it != vertices.end(); ++it)
  {
    Vertex vertex(mesh, *it);
    editor.addVertex(current_vertex++, vertex.point());
  }
  
  // Close editor
  editor.close();
}
//-----------------------------------------------------------------------------
SubMesh::~SubMesh()
{
  // Do nothing
}
//-----------------------------------------------------------------------------

