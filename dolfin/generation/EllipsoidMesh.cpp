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
#include <dolfin/geometry/ImplicitSurface.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include "ImplicitDomainMeshGenerator.h"
#include "EllipsoidMesh.h"

using namespace dolfin;

// Surface representation class
class EllipsoidSurface : public dolfin::ImplicitSurface
{
public:

  EllipsoidSurface(Point center, std::vector<double> dim)
    : ImplicitSurface(Sphere(center, *std::max_element(dim.begin(), dim.end()) ),
                        "manifold"), _center(center), _dim(dim) {}

  double operator()(const dolfin::Point& p) const
  {
    double d = 0.0;
    for (int i = 0; i < 3; ++i)
    {
      const double x = p[i] - _center[i];
      d += x*x/(_dim[i]*_dim[i]);
    }
    return d - 1.0;
  }

  const Point _center;
  const std::vector<double> _dim;
};

//-----------------------------------------------------------------------------
EllipsoidMesh::EllipsoidMesh(Point center, std::vector<double> dim,
                             std::size_t r)
{
  Timer timer("Generate ellipsoid mesh");

  // Create implicit representation of ellipsoid
  EllipsoidSurface surface(center, dim);

  // Generate mesh
  #ifdef HAS_CGAL
  double cell_size = 1.0/static_cast<double>(r);
  ImplicitDomainMeshGenerator::generate(*this, surface, cell_size) ;
  #else
  dolfin_error("Generation of ellipsoid meshes requires DOLFIN to be configured with CGAL");
  #endif
}
//-----------------------------------------------------------------------------
