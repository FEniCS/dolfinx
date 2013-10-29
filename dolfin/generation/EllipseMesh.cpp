// Copyright (C) 2013 Garth N. Wells
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
// First added:  2013-10-18
// Last changed:

#include <algorithm>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include "ImplicitDomainMeshGenerator.h"
#include "PolygonalMeshGenerator.h"
#include "EllipseMesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
EllipseMesh::EllipseMesh(Point center, std::vector<double> ellipse_dims,
                         double cell_size)
{
  Timer timer("Generate ellipse mesh");

  #ifdef HAS_CGAL
  dolfin_assert(ellipse_dims.size() == 2);

  // Estimate circumference
  const double a = ellipse_dims[0];
  const double b = ellipse_dims[1];

  // Compute number of points to place on boundary
  const double circumference = DOLFIN_PI*(3.0*(a + b)
    - std::sqrt(10.0*a*b - 3.0*(a*a + b*b)));
  const int num_points = 0.7*circumference/cell_size;

  // Create polyhedral representation
  std::vector<Point> polygon;
  for (int i = 0; i < num_points; ++i)
  {
    const double dtheta = 2.0*DOLFIN_PI/static_cast<double>(num_points);
    const double theta  = i*dtheta;
    const double x = a*std::cos(theta) + center.x();
    const double y = b*std::sin(theta) + center.y();
    polygon.push_back(Point(x, y));
  }
  polygon.push_back(polygon.front());

  // Generate mesh
  PolygonalMeshGenerator::generate(*this, polygon, cell_size);

  #else
  dolfin_error("EllipseMesh.cpp",
	       "generate ellipse mesh",
	       "Generation of ellipse meshes requires DOLFIN to be configured with CGAL");
  #endif
}
//-----------------------------------------------------------------------------
