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
      return x[0] >= 0.5;
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

  UnitSquare mesh(4, 4);

  // Create MeshFunction over facets
  MeshFunction<unsigned int> inflow_facets(mesh, mesh.topology().dim() - 1);
  inflow_facets.set_all(0);
  Inflow inflow;
  inflow.mark(inflow_facets, 1);

  // Create MeshFunction over cells
  MeshFunction<unsigned int> right_cells(mesh, mesh.topology().dim());
  right_cells.set_all(0);
  Right right;
  right.mark(right_cells, 1);

  // Plot
  plot(inflow_facets);
  plot(right_cells);

}
