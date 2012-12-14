// Copyright (C) 2012 Anders Logg
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
// First added:  2012-01-16
// Last changed: 2012-01-16

#include <cmath>

#include <dolfin/common/constants.h>
#include <dolfin/mesh/Mesh.h>
#include "MeshTransformation.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void MeshTransformation::rotate(Mesh& mesh, double angle, std::size_t axis)
{
  // Get mesh geometry
  MeshGeometry& geometry = mesh.geometry();
  const std::size_t gdim = geometry.dim();

  // Reset center of mass
  std::vector<double> c(gdim);
  for (std::size_t j = 0; j < gdim; j++)
    c[j] = 0.0;

  // Sum all vertex coordinates
  for (std::size_t i = 0; i < geometry.size(); i++)
  {
    const double* x = geometry.x(i);
    for (std::size_t j = 0; j < gdim; j++)
      c[j] += x[j];
  }

  // Divide by the number of vertices
  for (std::size_t j = 0; j < gdim; j++)
    c[j] /= static_cast<double>(geometry.size());

  // Set up point
  dolfin_assert(gdim <= 3);
  Point p;
  for (std::size_t j = 0; j < gdim; j++)
    p[j] = c[j];

  // Rotate around center of mass
  rotate(mesh, angle, axis, p);
}
//-----------------------------------------------------------------------------
void MeshTransformation::rotate(Mesh& mesh, double angle, std::size_t axis,
                                const Point& p)
{
  // Compute angle (radians)
  const double theta = angle / 180.0 * DOLFIN_PI;

  // Get coordinates of point
  const double* c = p.coordinates();

  // Check dimension
  const std::size_t gdim = mesh.geometry().dim();
  if (gdim == 2)
  {
    // Check axis of rotation (must be 2)
    if (axis != 2)
    {
      dolfin_error("MeshTransformation.cpp",
                   "rotate mesh",
                   "A 2D mesh can only be rotated around the z-axis (axis = 2)");
    }

    // Set up rotation matrix
    const double S00 = cos(theta); const double S01 = -sin(theta);
    const double S10 = sin(theta); const double S11 = cos(theta);

    // Rotate all points
    MeshGeometry& geometry = mesh.geometry();
    for (std::size_t i = 0; i < geometry.size(); i++)
    {
      // Get coordinate
      double* x = geometry.x(i);

      // Compute vector from rotation point
      const double dx0 = x[0] - c[0];
      const double dx1 = x[1] - c[1];

      // Rotate
      const double x0 = c[0] + S00*dx0 + S01*dx1;
      const double x1 = c[1] + S10*dx0 + S11*dx1;

      // Store coordinate
      x[0] = x0;
      x[1] = x1;
    }
  }
  else if (gdim == 3)
  {
    // Set up 2D rotation matrix
    const double S00 = cos(theta); const double S01 = -sin(theta);
    const double S10 = sin(theta); const double S11 = cos(theta);

    // Initialize 3D rotation matrix to identity matrix
    double R00 = 1.0; double R01 = 0.0; double R02 = 0.0;
    double R10 = 0.0; double R11 = 1.0; double R12 = 0.0;
    double R20 = 0.0; double R21 = 0.0; double R22 = 1.0;

    // Set up 3D rotation matrix
    switch (axis)
    {
    case 0:
      R11 = S00; R12 = S01; R21 = S10; R22 = S11;
      break;
    case 1:
      R00 = S00; R02 = S01; R20 = S10; R22 = S11;
      break;
    case 2:
      R00 = S00; R01 = S01; R10 = S10; R11 = S11;
      break;
    default:
      dolfin_error("MeshTransformation.cpp",
                   "rotate mesh",
                   "A 3D mesh can only be rotated around axis 0, 1 or 2");
    }

    // Rotate all points
    MeshGeometry& geometry = mesh.geometry();
    for (std::size_t i = 0; i < geometry.size(); i++)
    {
      // Get coordinate
      double* x = geometry.x(i);

      // Compute vector from rotation point
      const double dx0 = x[0] - c[0];
      const double dx1 = x[1] - c[1];
      const double dx2 = x[2] - c[2];

      // Rotate
      const double x0 = c[0] + R00*dx0 + R01*dx1 + R02*dx2;
      const double x1 = c[1] + R10*dx0 + R11*dx1 + R12*dx2;
      const double x2 = c[2] + R20*dx0 + R21*dx1 + R22*dx2;

      // Store coordinate
      x[0] = x0;
      x[1] = x1;
      x[2] = x2;
    }
  }
  else
  {
    dolfin_error("MeshTransformation.cpp",
                 "rotate mesh",
                 "Mesh rotation has not been implemented for meshes of dimension %d",
                 gdim);
  }
}
//-----------------------------------------------------------------------------
