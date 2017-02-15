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
// Last changed: 2017-02-15

#include <sstream>
#include <iomanip>
#include <dolfin/log/log.h>
#include <dolfin/log/LogStream.h>
#include "GeometryDebugging.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void GeometryDebugging::print(std::vector<Point> simplex)
{
  set_indentation_level(0);
  cout << "Simplex: " << simplex2string(simplex, true) << endl;
}
//-----------------------------------------------------------------------------
void GeometryDebugging::print(std::vector<Point> simplex_0,
                              std::vector<Point> simplex_1)
{
  set_indentation_level(0);
  cout << "Simplex 0: " << simplex2string(simplex_0, true) << endl;
  cout << "Simplex 1: " << simplex2string(simplex_1, true) << endl;
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
std::string GeometryDebugging::point2string(const Point& p)
{
  std::stringstream s;
  s << std::setprecision(17);
  s << "[" << p.x() << "," << p.y() << "," << p.z() << "]";
  return s.str();
}
//-----------------------------------------------------------------------------
std::string GeometryDebugging::simplex2string(const std::vector<Point>& simplex,
                                              bool rowmajor)
{
  std::size_t n = simplex.size();
  std::stringstream s;
  s << std::setprecision(17);

  if (rowmajor)
  {
    s << "[";
    for (std::size_t i = 0; i < n - 1; i ++)
    {
      s << point2string(simplex[i]);
      s << ",";
    }
    s << point2string(simplex[n - 1]) << "]";
  }
  else
  {
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
  }

  return s.str();
}
//-----------------------------------------------------------------------------
