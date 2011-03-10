// Copyright (C) 2011 Marie E. Rognes
// Licensed under the GNU LGPL Version 3 or any later version
//
// Last changed: 2011-03-10

#include <dolfin.h>

using namespace dolfin;

int main()
{
  // Sub domain for right part of mesh
  class Right : public SubDomain
  {
    bool inside(const Array<double>& x, bool on_boundary) const
    {
      return x[0] <= 0.5;
    }
  };

  // Sub domain for inflow (right)
  class Inflow : public SubDomain
  {
    bool inside(const Array<double>& x, bool on_boundary) const
    {
      return x[0] > 1.0 - DOLFIN_EPS && on_boundary;
    }
  };

  UnitSquare mesh(1, 1);

  // Create MeshFunction over facets
  MeshFunction<unsigned int> inflow_facets(mesh, mesh.topology().dim() - 1, 0);
  Inflow inflow;
  inflow.mark(inflow_facets, 1);
  //plot(inflow_facets);

  // Create MeshFunction over cells
  MeshFunction<unsigned int> right_cells(mesh, mesh.topology().dim(), 0);
  Right right;
  right.mark(right_cells, 1);

  // Copy data over to mesh data (Better way?)
  MeshFunction<unsigned int>* materials = \
    mesh.data().create_mesh_function("material indicators", mesh.topology().dim());
  for (CellIterator c(mesh); !c.end(); ++c)
    (*materials)[*c] = right_cells[*c];

  // Mark cells for refinement
  MeshFunction<bool> cell_markers(mesh, mesh.topology().dim(), false);
  Point p(1.0, 0.0);
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    if (c->midpoint().distance(p) < 0.5)
      cell_markers[*c] = true;
  }

  // Refine mesh -> new_mesh
  Mesh new_mesh = refine(mesh, cell_markers);
  plot(new_mesh);

  // Extract and plot refined material indicators
  MeshFunction<dolfin::uint>* new_materials =                   \
    new_mesh.data().mesh_function("material indicators");
  plot(*new_materials);

  return 0;
}
