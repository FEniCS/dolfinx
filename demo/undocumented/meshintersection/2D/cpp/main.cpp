// Copyright (C) 2008 Kristoffer Selim
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
// Modified by Benjain Kehlet 2012
//
// First added:  2010-01-14
// Last changed: 2012-07-19

#include <dolfin.h>
#include <math.h>

using namespace dolfin;

#ifdef HAS_CGAL

int main()
{
  // Create meshes (omega0 overlapped by omega1)
  UnitCircle omega0(20);
  UnitSquare omega1(20, 20);

  boost::shared_ptr<dolfin::MeshFunction<std::size_t> >intersection1(new dolfin::MeshFunction<std::size_t>(omega0, omega0.topology().dim()));

  VTKPlotter p(intersection1);
  p.parameters["rescale"] = true;
  p.parameters["wireframe"] = false;
  // p.parameters["axes"] = true;
  p.parameters["scalarbar"] = false;

  double polygon[] = { 0.0, 0.0,
                       1.0, 0.0,
                       1.0, 1.0,
                       0.0, 1.0,
                       0.0, 0.0 };
  Array<double> _polygon(10, polygon);
  p.add_polygon(_polygon);

  {
    // Access mesh geometry
    MeshGeometry& geometry = omega0.geometry();

    // Move and scale circle
    for (VertexIterator vertex(omega0); !vertex.end(); ++vertex)
    {
      double* x = geometry.x(vertex->index());
      x[0] = 0.5*x[0] + 1.0;
      x[1] = 0.5*x[1] + 1.0;
    }

    // Iterate over angle
    const double dtheta = 0.1*DOLFIN_PI;
    for (double theta = 0; theta < 2*DOLFIN_PI; theta += dtheta)
    {
      // Compute intersection with boundary of square
      BoundaryMesh boundary(omega1);
      std::set<dolfin::uint> cells;
      omega0.intersected_cells(boundary, cells);

      // Copy values to mesh function for plotting
      *intersection1 = 0;
      for (std::set<dolfin::uint>::const_iterator i = cells.begin(); i != cells.end(); i++)
        (*intersection1)[*i] = 1;

      // Plot intersection
      p.plot();

      // Rotate circle around (0.5, 0.5)
      for (VertexIterator vertex(omega0); !vertex.end(); ++vertex)
      {
        double* x = geometry.x(vertex->index());
        const double xr = x[0] - 0.5;
        const double yr = x[1] - 0.5;
        x[0] = 0.5 + (cos(dtheta)*xr - sin(dtheta)*yr);
        x[1] = 0.5 + (sin(dtheta)*xr + cos(dtheta)*yr);
      }

      // Clear the cached intersection operator. Necessary because mesh
      // has changed.
      omega0.intersection_operator().clear();
    }
  }


  // Repeat the same with the rotator in the cavity example.
  Rectangle background_mesh(-2.0, -2.0, 2.0, 2.0, 30, 30);
  boost::shared_ptr<dolfin::MeshFunction<std::size_t> >intersection2(new dolfin::MeshFunction<std::size_t>(background_mesh, background_mesh.topology().dim()));

  VTKPlotter p2(intersection2);
  p2.parameters["rescale"] = true;
  p2.parameters["wireframe"] = true;
  // p.parameters["axes"] = true;
  p2.parameters["scalarbar"] = false;

  {
    Mesh structure_mesh("../rotator.xml.gz");

    // Access mesh geometry
    MeshGeometry& geometry = structure_mesh.geometry();

    // Iterate over angle
    double theta = 0.0;
    double dtheta = 0.1*DOLFIN_PI;
    while (theta < 2*DOLFIN_PI + dtheta)
    {
      std::set<dolfin::uint> cells;
      background_mesh.intersected_cells(structure_mesh, cells);

      // Mark intersected values
      *intersection2 = 0;

      // Copy values to mesh function for plotting
      for (std::set<dolfin::uint>::const_iterator i = cells.begin(); i != cells.end(); i++)
        (*intersection2)[*i] = 1;

      p2.plot();

      // Rotate rotator
      for (VertexIterator vertex(structure_mesh); !vertex.end(); ++vertex)
      {
        double* x = geometry.x(vertex->index());
        const double xr = x[0];
        const double yr = x[1];
        x[0] = cos(dtheta)*xr - sin(dtheta)*yr;
        x[1] = sin(dtheta)*xr + cos(dtheta)*yr;
      }

      // Clear the cached intersection operator. Necessary because mesh
      // has changed.
      structure_mesh.intersection_operator().clear();

      theta += dtheta;
    }
  }

  interactive();
  return 0;
}

#else

int main()
{
  info("DOLFIN must be compiled with CGAL to run this demo.");
  return 0;
}

#endif
