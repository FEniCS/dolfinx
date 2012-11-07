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
// First added:  2012-10-31
// Last changed: 2012-10-31

// This class is capable of parsing surface file to a cgal polyhedron
// .off files are parsed by cgal


#ifndef __SURFACE_FILE_READER_H
#define __SURFACE_FILE_READER_H

#include "cgal_csg3d.h"

namespace dolfin
{
  namespace csg
  {
    class SurfaceFileReader 
    {
    public:
      static void readSurfaceFile(std::string filename, Exact_Polyhedron_3& p);
      static void readSTLFile(std::string filename, Exact_Polyhedron_3& p);
      static bool has_self_intersections(Exact_Polyhedron_3& p);
      static CGAL::Bbox_3 getBoundingBox(csg::Polyhedron_3& polyhedron);
      static double getBoundingSphereRadius(csg::Polyhedron_3& polyhedron);
    };
  }
}
#endif
