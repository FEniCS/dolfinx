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
// First added:  2012-12-01
// Last changed: 2012-12-01

#include <iostream>
#include <fstream>

#include <dolfin/common/MPI.h>
#include <dolfin/log/log.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Edge.h>
#include "SVGFile.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
SVGFile::SVGFile(const std::string filename): GenericFile(filename, "SVG")
{
  // Do nothing
}
//-----------------------------------------------------------------------------
SVGFile::~SVGFile()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void SVGFile::write(const Mesh& mesh)
{
  // Get file path and generate filename
  std::string prefix;
  prefix.assign(_filename, 0, _filename.find_last_of("."));
  std::ostringstream _filename;
  _filename << prefix;
  if (MPI::size(mesh.mpi_comm()) > 1)
    _filename << "_p" << MPI::rank(mesh.mpi_comm());
  _filename << ".svg";

  // Open file
  std::ofstream fp(_filename.str().c_str(), std::ios_base::out);

  // Get parameters
  const double relative_line_width = parameters["relative_line_width"];

  // Get dimensions
  double xmin = std::numeric_limits<double>::max();
  double xmax = std::numeric_limits<double>::min();
  double ymin = std::numeric_limits<double>::max();
  double ymax = std::numeric_limits<double>::min();
  const std::size_t gdim = mesh.geometry().dim();
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    xmin = std::min(xmin, v->x(0));
    xmax = std::max(xmax, v->x(0));
    if (gdim > 1)
    {
      ymin = std::min(ymin, v->x(1));
      ymax = std::max(ymax, v->x(1));
    }
  }
  if (gdim == 1)
  {
    ymin = xmin;
    ymax = xmax;
  }

  // Write header
  fp << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>" << std::endl;
  fp << "<!-- Created with DOLFIN (http://fenicsproject.org/) -->" << std::endl;
  fp << std::endl;
  fp << "<svg"
     << " x=\"" << xmin << "\""
     << " y=\"" << ymin << "\""
     << " width=\""  << (xmax - xmin) << "\""
     << " height=\"" << (ymax - ymin) << "\""
     << ">" << std::endl;
  fp << "  <g id=\"layer1\">" << std::endl;

  // Write edges of mesh
  for (EdgeIterator e(mesh); !e.end(); ++e)
  {
    // Get the two vertex coordinates
    Vertex v0(mesh, e->entities(0)[0]);
    Vertex v1(mesh, e->entities(0)[1]);
    double x0(v0.x(0));
    double x1(v1.x(0));
    double y0(0);
    double y1(0);
    if (gdim > 1)
    {
      y0 = v0.x(1);
      y1 = v1.x(1);
    }

    // Write SVG data
    fp << "    <path style=\"fill:none;stroke:#000000;stroke-width:"
       << (relative_line_width*e->length())
       << "px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1\""
       << std::endl
       << "          d=\"M "
       << x0 << "," << y0 << " " << x1 << "," << y1 << "\""
       << " id=\"edge" << e->index() << "\"/>" << std::endl;
  }

  // Write footer
  fp << "  </g>" << std::endl;
  fp << "</svg>" << std::endl;

  // Close file
  fp.close();
}
//-----------------------------------------------------------------------------
