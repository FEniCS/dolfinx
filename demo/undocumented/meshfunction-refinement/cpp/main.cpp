// Copyright (C) 2011 Marie E. Rognes
// Licensed under the GNU LGPL Version 3 or any later version
//
// Last changed: 2011-03-12

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

  UnitSquare mesh(5, 5);

  // Create MeshFunction over cells
  MeshFunction<unsigned int> right_cells(mesh, mesh.topology().dim(), 0);
  Right right;
  right.mark(right_cells, 1);

  // Create MeshFunction over facets
  MeshFunction<unsigned int> inflow_facets(mesh, mesh.topology().dim() - 1, 0);
  Inflow inflow;
  inflow.mark(inflow_facets, 1);

  // Mark cells for refinement
  MeshFunction<bool> cell_markers(mesh, mesh.topology().dim(), false);
  Point p(1.0, 0.0);
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    if (c->midpoint().distance(p) < 0.5)
      cell_markers[*c] = true;
  }

  // Refine mesh
  adapt(mesh, cell_markers);

  // Adapt cell function to refined mesh
  adapt(right_cells, mesh.child_shared_ptr());

  // Adapt facet function to refined mesh
  adapt(inflow_facets, mesh.child_shared_ptr());

  return 0;
}
