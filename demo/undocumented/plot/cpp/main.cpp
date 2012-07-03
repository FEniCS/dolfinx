// Copyright (C) 2006-2007 Anders Logg
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
// Modified by Benjamin Kehlet 2012
//
// First added:  2007-05-29
// Last changed: 2012-06-25
//
// This demo illustrates basic plotting.

#include <dolfin.h>

using namespace dolfin;

class ScalarExpression : public Expression
{
public:

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = t * 100 * exp(-10.0 * (pow(x[0] - t, 2) + pow(x[1] - t, 2)));
  }

  double t;

};

class VectorExpression : public Expression
{
public:

  VectorExpression() : Expression(2) {}

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = -(x[1] - t)*exp(-10.0*(pow(x[0] - t, 2) + pow(x[1] - t, 2)));
    values[1] =  (x[0] - t)*exp(-10.0*(pow(x[0] - t, 2) + pow(x[1] - t, 2)));
  }

  double t;

};

int main()
{
  // Read mesh from file
  Mesh mesh("../dolfin-2.xml.gz");

  // Have some fun with the mesh
  const double R = 0.15;
  const double H = 0.025;
  double X = 0.3;
  double Y = 0.4;
  double dX = H;
  double dY = 1.5*H;
  double* coordinates = mesh.coordinates();
  const std::vector<double> original(coordinates, coordinates + 2*mesh.num_vertices());

  for (dolfin::uint i = 0; i < 100; i++)
  {
    if (X < H || X > 1.0 - H)
      dX = -dX;

    if (Y < H || Y > 1.0 - H)
      dY = -dY;

    X += dX;
    Y += dY;

    for (dolfin::uint j = 0; j < mesh.num_vertices(); ++j)
    {
      const double x = coordinates[2*j];
      const double y = coordinates[2*j + 1];
      const double r = std::sqrt((x - X)*(x - X) + (y - Y)*(y - Y));
      if (r < R)
      {
        coordinates[2*j]     = X + (r/R)*(r/R)*(x - X);
        coordinates[2*j + 1] = Y + (r/R)*(r/R)*(y - Y);
      }
    }

    plot(mesh, "Plotting mesh");

    std::copy(original.begin(), original.end(), coordinates);
  }

  // Plot scalar function
  Parameters p;
  p.add("rescale", true);
  p.add("title", "Plotting scalar function");

  ScalarExpression f_scalar;

  // FIXME: VTK sets the center and zoom incorrectly if the loop starts at 0.0
  for (double t = 0.01; t < 1.0; t += 0.01)
  {
    f_scalar.t = t;
    plot(f_scalar, mesh, p);
  }

  // Plot vector function
  UnitSquare unit_square(16, 16);
  VectorExpression f_vector;
  for (double t = 0.0; t < 1.0; t += 0.005)
  {
    f_vector.t = t;
    plot(f_vector, unit_square, "Plotting vector function");
  }

  interactive();

  return 0;
}
