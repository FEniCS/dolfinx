// Copyright (C) 2016 Anders Logg
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
// First added:  2016-05-05
// Last changed: 2016-05-05

#include <sstream>
#include <dolfin/log/log.h>
#include <dolfin/log/LogStream.h>
#include "GeometryDebugging.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void GeometryDebugging::print(std::vector<Point> simplex)
{
  set_indentation_level(0);
  cout << "Simplex:";
  for (const Point p : simplex)
    cout << " " << p;
  cout << endl;
}
//-----------------------------------------------------------------------------
void GeometryDebugging::print(std::vector<Point> simplex_0,
                              std::vector<Point> simplex_1)
{
  set_indentation_level(0);
  cout << "Simplex 0:";
  for (const Point p : simplex_0)
    cout << "-" << point2string(p);
  cout << endl;

  cout << "Simplex 1:";
  for (const Point p : simplex_1)
    cout << "-" << point2string(p);
  cout << endl;
}
//-----------------------------------------------------------------------------
void GeometryDebugging::plot(std::vector<Point> simplex)
{
  set_indentation_level(0);
  init_plot();

  cout << "# Plot simplex" << endl;
  cout << "ax.plot_trisurf(" << simplex2string(simplex) << ")" << endl;
  cout << "pl.show()" << endl;
  cout << endl;
}
//-----------------------------------------------------------------------------
void GeometryDebugging::plot(std::vector<Point> simplex_0,
                             std::vector<Point> simplex_1)
{
  set_indentation_level(0);
  init_plot();

  cout << "# Plot simplex intersection" << endl;
  cout << "ax.plot_trisurf(" << simplex2string(simplex_0) << ", color='r')" << endl;
  cout << "ax.plot_trisurf(" << simplex2string(simplex_1) << ", color='b')" << endl;
  cout << "pl.show()" << endl;
  cout << endl;
}
//-----------------------------------------------------------------------------
void GeometryDebugging::init_plot()
{
  set_indentation_level(0);
  cout << "# Initialize matplotlib 3D plotting" << endl;
  cout << "from mpl_toolkits.mplot3d import Axes3D" << endl;
  cout << "import matplotlib.pyplot as pl" << endl;
  cout << "ax = pl.figure().gca(projection='3d')" << endl;
  cout << endl;
}
//-----------------------------------------------------------------------------
std::string GeometryDebugging::point2string(Point p)
{
  std::stringstream s;
  s << "(" << p.x() << "," << p.y() << "," << p.z() << ")";
  return s.str();
}
//-----------------------------------------------------------------------------
std::string GeometryDebugging::simplex2string(std::vector<Point> simplex)
{
  std::size_t n = simplex.size();
  std::stringstream s;
  s << "[";
  for (std::size_t i = 0; i < n - 1; i++)
    s << simplex[i].x() << ",";
  s << simplex[n - 1].x() << "]";
  s << ",";
  s << "[";
  for (std::size_t i = 0; i < n - 1; i++)
    s << simplex[i].y() << ",";
  s << simplex[n - 1].y() << "]";
  s << ",";
  s << "[";
  for (std::size_t i = 0; i < n - 1; i++)
    s << simplex[i].z() << ",";
  s << simplex[n - 1].z() << "]";

  return s.str();
}
//-----------------------------------------------------------------------------
