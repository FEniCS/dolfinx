// Copyright (C) 2012 Benjamin Kehlet
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
// Modified by Johannes Ring, 2012
//
// First added:  2012-05-12
// Last changed: 2012-12-20

#include <cmath>

#include "CSGGeometries3D.h"
#include "CSGGeometry.h"
#include "CSGPrimitives3D.h"
#include "CSGOperators.h"

using namespace dolfin;

boost::shared_ptr<CSGGeometry> CSGGeometries::lego( std::size_t n0, std::size_t n1, std::size_t n2, double x0, double x1, double x2 )
{
  // Standard dimensions for LEGO bricks / m
  const double P = 8.0 * 0.001;
  const double h = 3.2 * 0.001;
  const double D = 5.0 * 0.001;
  const double b = 1.7 * 0.001;
  const double d = 0.2 * 0.001;

  // Create brick
  boost::shared_ptr<CSGGeometry> lego(new Box(x0 + 0.5*d, x1 + 0.5*d, x2,
						x0 + n0*P - 0.5*d, x1 + n1*P - 0.5*d, x2 + n2*h));

  // Add knobs
  for (std::size_t i = 0; i < n0; i++)
  {
    for (std::size_t j = 0; j < n1; j++)
    {
      const double x = x0 + (i + 0.5)*P;
      const double y = x1 + (j + 0.5)*P;
      const double z = x2;

      boost::shared_ptr<CSGGeometry> knob(new Cone(Point(x, y, z),
							     Point(x, y, z + n2*h + b),
							     0.5*D, 0.5*D));
      lego = lego + knob;
    }
  }

  return lego;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<CSGGeometry> CSGGeometries::propeller(double r, double R, double w, double h )
{
  // Parameters
  //const double v = 30;     // initial rotation of blades
  //const double u = 20;     // additional rotation of blades
  const double l = 0.8*w;  // length of cone
  const double a = h;      // radius of tip of cone

  // // Create blades
  boost::shared_ptr<CSGGeometry> blade_0( new Box(0.8*r,  -0.5*h, -0.5*w,     R,   0.5*h, 0.5*w));
  boost::shared_ptr<CSGGeometry> blade_1( new Box(-R,     -0.5*h, -0.5*w, -0.8*r,  0.5*h, 0.5*w));
  boost::shared_ptr<CSGGeometry> blade_2( new Box(-0.5*h,  0.8*r, -0.5*w,  0.5*h,      R, 0.5*w));
  boost::shared_ptr<CSGGeometry> blade_3( new Box(-0.5*h,     -R, -0.5*w,  0.5*h, -0.8*r, 0.5*w));

  // // Rotate blades
  // // blade_0.rotate(-v, 0);
  // // blade_1.rotate(v, 0);
  // // blade_2.rotate(v, 1);
  // // blade_3.rotate(-v, 1);

  // Create blade tips
  boost::shared_ptr<CSGGeometry> blade_tip_0(new Cylinder( Point( R,     -0.5*h, 0), Point( R,  0.5*h, 0), 0.5*w));
  boost::shared_ptr<CSGGeometry> blade_tip_1(new Cylinder( Point(-R,     -0.5*h, 0), Point(-R,  0.5*h, 0), 0.5*w));
  boost::shared_ptr<CSGGeometry> blade_tip_2(new Cylinder( Point(-0.5*h,  R,     0), Point( 0.5*h,  R, 0), 0.5*w));
  boost::shared_ptr<CSGGeometry> blade_tip_3(new Cylinder( Point(-0.5*h, -R,     0), Point( 0.5*h, -R,     0), 0.5*w));

  // // Rotate blade tips
  // // blade_tip_0.rotate(-v, 0);
  // // blade_tip_1.rotate(v, 0);
  // // blade_tip_2.rotate(v, 1);
  // // blade_tip_3.rotate(-v, 1);

  // Add blade tips
  blade_0 = blade_0 + blade_tip_0;
  blade_1 = blade_1 + blade_tip_1;
  blade_2 = blade_2 + blade_tip_2;
  blade_3 = blade_3 + blade_tip_3;

  // // Add blades
  boost::shared_ptr<CSGGeometry> blades = blade_0 + blade_1 + blade_2 + blade_3;

  // Create outer cylinder
  boost::shared_ptr<CSGGeometry> cylinder_outer(new Cylinder(Point(0, 0, -0.5*w), Point(0, 0, 0.5*w), r));

  // Create inner cylinder
  boost::shared_ptr<CSGGeometry> cylinder_inner(new Cylinder(Point(0, 0, -0.5*w), Point(0, 0, 0.5*w), 0.5*r));

  // Create center cone
  boost::shared_ptr<CSGGeometry> cone( new Cone(Point(0, 0, -0.5*w), Point(0, 0, -0.5*w - l), r, a));

  // Create sphere for tip of cone
  const double d = a*(r - a) / l;
  boost::shared_ptr<CSGGeometry> tip(new Sphere(Point(0, 0, -0.5*w - l + d), sqrt(a*a + d*d)));

  // Build propeller from parts
  return cylinder_outer - cylinder_inner + cone + tip + blades;
}
