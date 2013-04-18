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
// Modified by Benjamin Kehlet, 2012
// Modified by Johannes Ring, 2012
//
// First added:  2012-04-11
// Last changed: 2012-11-05

#ifndef __CSG_GEOMETRY_H
#define __CSG_GEOMETRY_H

#include <cstddef>
#include <vector>
#include <list>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/Variable.h>

namespace dolfin
{


  /// Geometry described by Constructive Solid Geometry (CSG)

  class CSGGeometry : public Variable
  {
  public:

    /// Constructor
    CSGGeometry();

    /// Destructor
    virtual ~CSGGeometry();

    /// Return dimension of geometry
    virtual std::size_t dim() const = 0;

    /// Informal string representation
    virtual std::string str(bool verbose) const = 0;

    void set_subdomain(std::size_t i, boost::shared_ptr<CSGGeometry> s);
    void set_subdomain(std::size_t i, CSGGeometry& s);

    enum Type { Box, Sphere, Cone, Tetrahedron, Surface3D, Circle, Ellipse, Rectangle, Polygon, Union, Intersection, Difference };
    virtual Type getType() const = 0;

    virtual bool is_operator() const = 0;

    std::list<std::pair<std::size_t, boost::shared_ptr<const CSGGeometry> > > subdomains;
  };

  

}

#endif
