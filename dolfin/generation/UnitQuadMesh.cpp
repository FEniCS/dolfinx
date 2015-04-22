
#include <dolfin/common/constants.h>
#include <dolfin/mesh/MeshEditor.h>
#include "UnitQuadMesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
UnitQuadMesh::UnitQuadMesh(std::size_t nx, std::size_t ny)
{

  MeshEditor editor;
  editor.open(*this, CellType::quadrilateral, 2, 2);

  // Create vertices and cells:
  editor.init_vertices_global((nx + 1)*(ny + 1), (nx + 1)*(ny + 1));
  editor.init_cells_global(nx*ny, nx*ny);

  // Storage for vertices
  std::vector<double> x(2);

  const double a = 0.0;
  const double b = 1.0;
  const double c = 0.0;
  const double d = 1.0;

  // Create main vertices:
  std::size_t vertex = 0;
  for (std::size_t iy = 0; iy <= ny; iy++)
  {
    x[1] = c + ((static_cast<double>(iy))*(d - c)/static_cast<double>(ny));
    for (std::size_t ix = 0; ix <= nx; ix++)
    {
      x[0] = a + ((static_cast<double>(ix))*(b - a)/static_cast<double>(nx));
      editor.add_vertex(vertex, x);
      vertex++;
    }
  }

  // Create rectangles
  std::size_t cell = 0;
  std::vector<std::size_t> v(4);
  for (std::size_t iy = 0; iy < ny; iy++)
    for (std::size_t ix = 0; ix < nx; ix++)
    {
      v[0] = iy*(nx + 1) + ix;
      v[1] = v[0] + 1;
      v[2] = v[0] + (nx + 1);
      v[3] = v[1] + (nx + 1);
      editor.add_cell(cell, v);
      ++cell;
    }

  // Close mesh editor
  editor.close();
}

//-----------------------------------------------------------------------------
