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
// Last changed: 2012-05-14

#ifndef __CSG_GEOMETRIES_H
#define __CSG_GEOMETRIES_H

#include <boost/shared_ptr.hpp>
#include <dolfin/geometry/Point.h>
#include "CSGGeometry.h"

namespace dolfin
{
  class CSGGeometries
  {
  public:

    // A standard LEGO brick starting at the point x with (n0, n1)
    // knobs and height n2. The height should be 1 for a thin brick or 3
    // for a regular brick.
    static boost::shared_ptr<CSGGeometry> lego(std::size_t n0, std::size_t n1,
                                               std::size_t n2, double x0,
                                               double x1, double x2);

    // A simple propeller with parameters r - radius of center body, R - length of blades,
    // w - width of blades and h - thicknes of blades
    static boost::shared_ptr<CSGGeometry> propeller(double r=0.125,
                                                    double R=0.5,
                                                    double w=0.3,
                                                    double h=0.025 );
  };
}

#endif
