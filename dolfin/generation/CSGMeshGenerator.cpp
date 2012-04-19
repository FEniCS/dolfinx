// Copyright (C) 2012 Anders Logg (and others, add authors)
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
// First added:  2012-01-01
// Last changed: 2012-04-19

#include <dolfin/log/log.h>
#include "CSGMeshGenerator.h"
#include "CSGGeometry.h"

using namespace dolfin;
#ifdef HAS_CGAL
//-----------------------------------------------------------------------------
void CSGMeshGenerator::generate(Mesh& mesh,
                                const CSGGeometry& geometry)
{
  // Temporary implementation just to generate something
  if (geometry.dim() == 2)
  {
    generate_2d(mesh, geometry);
  }
  else if (geometry.dim() == 3)
  {
    generate_3d(mesh, geometry);
  }
  else
  {
    dolfin_error("CSGMeshGenerator.cpp",
                 "create mesh from CSG geometry",
                 "Unhandled geometry dimension %d", geometry.dim());
  }
}
//-----------------------------------------------------------------------------
void CSGMeshGenerator::generate_2d(Mesh& mesh,
                                const CSGGeometry& geometry)
{

}
//-----------------------------------------------------------------------------
void CSGMeshGenerator::generate_3d(Mesh& mesh,
                                const CSGGeometry& geometry)
{
  Nef_polyhedron_3 cgal_geometry = geometry.get_cgal_type_3D();

  
}
//-----------------------------------------------------------------------------
#else
void CSGMeshGenerator::generate(Mesh& mesh,
                                const CSGGeometry& geometry)
{
  dolfin_error("CSGMeshGenerator.cpp",
	       "create mesh from CSG geometry",
	       "Mesh generation not available. Dolfin has been compiled without CGAL.");
}
#endif
