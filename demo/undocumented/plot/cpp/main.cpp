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
// Modified by Benjamin Kehlet, 2012
//
// First added:  2007-05-29
// Last changed: 2012-06-25

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


class VectorValuedExpression : public Expression
{
 public:
  VectorValuedExpression() : Expression(2) {}
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = -(x[1] - t)*exp(-10.0*(pow(x[0] - t, 2) + pow(x[1] - t, 2)));
    values[1] =  (x[0] - t)*exp(-10.0*(pow(x[0] - t, 2) + pow(x[1] - t, 2)));
  }

  double t;
};



int main()
{
  // Read and plot mesh from file
  Mesh mesh("../dolfin-2.xml.gz");

  

  double R = 0.15;
  double H = 0.025;
  double X = 0.3;
  double Y = 0.4;
  double dX = H;
  double dY = 1.5*H;
  double* coordinates = mesh.coordinates();
  double original[mesh.num_vertices()*2];
  memcpy(original, coordinates, sizeof(original));

  double x, y, r;
    
  for(int i = 0; i < 100; i++) 
  {
    if (X < H || X > 1.0 - H) 
    {
      dX = -dX;
    }
    
    if (Y < H || Y > 1.0 - H) 
    {
      dY = -dY;
    }
    
    X += dX;
    Y += dY;
    
    for (int j = 0; j < mesh.num_vertices(); ++j) 
    {
      x = coordinates[2*j];
      y = coordinates[2*j+1];
      r = sqrt(pow((x - X),2) + pow((y - Y),2));
      if (r < R) 
      {
  	coordinates[2*j]   = X + pow((r/R),2)*(x - X);
  	coordinates[2*j+1] = Y + pow((r/R),2)*(y - Y);
      }
    }
    plot(mesh, "Animated mesh plot");
    
    for (int j = 0; j < mesh.num_vertices()*2; ++j) 
    {
      coordinates[j] = original[j];
    }
  }

  

  // Plot scalar function
  Parameters p;
  p.add("rescale", true);

  ScalarExpression f_scalar;

  // FIXME: VTK sets the center and zoom incorrectly
  // if the loop starts at 0.0
  for(double t = 0.01; t < 1.0; t += 0.01)
  {
    f_scalar.t = t;
    plot(f_scalar, mesh, p);
  }

  
  // Plot vector function
  UnitSquare unit_square(16, 16);
  VectorValuedExpression f_vectorvalued;
  for (double t = 0.0; t < 1.0; t += 0.005)
  {
    f_vectorvalued.t = t;
    plot(f_vectorvalued, unit_square);
  }
  

  interactive();
}
