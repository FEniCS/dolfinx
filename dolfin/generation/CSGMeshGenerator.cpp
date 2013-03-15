// Copyright (C) 2012 Anders Logg, Benjamin Kehlet, Johannes Ring
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
// Modified by Joachim B Haga 2012
//
// First added:  2012-01-01
// Last changed: 2013-03-15


#include <dolfin/log/log.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include "CSGMeshGenerator.h"
#include "CSGGeometry.h"
#include "CSGCGALMeshGenerator2D.h"
#include "CSGCGALMeshGenerator3D.h"

using namespace dolfin;

#ifdef HAS_CGAL

//-----------------------------------------------------------------------------
void CSGMeshGenerator::generate(Mesh& mesh,
                                const CSGGeometry& geometry,
                                std::size_t resolution)
{
  if (geometry.dim() == 2)
  {
    CSGCGALMeshGenerator2D generator(geometry);
    generator.parameters["mesh_resolution"] = static_cast<int>(resolution);
    generator.generate(mesh);
  }
  else if (geometry.dim() == 3)
  {
    CSGCGALMeshGenerator3D generator(geometry);
    generator.parameters["mesh_resolution"] = static_cast<int>(resolution);
    generator.generate(mesh);
  }
  else
  {
    dolfin_error("CSGMeshGenerator.cpp",
                 "create mesh from CSG geometry",
                 "Unhandled geometry dimension %d", geometry.dim());
  }
}
//-----------------------------------------------------------------------------
void CSGMeshGenerator::generate(BoundaryMesh& mesh,
                                const CSGGeometry& geometry)
{
  if (geometry.dim() == 2)
  {
    // Generate boundary mesh directly from 2D CSGGeometry is not
    // implemented. Instead, generate the full mesh and extract its boundary.
    CSGCGALMeshGenerator2D generator(geometry);
    Mesh full_mesh;
    generator.generate(full_mesh);

    mesh = BoundaryMesh(full_mesh, "exterior");
  }
  else if (geometry.dim() == 3)
  {
    CSGCGALMeshGenerator3D generator(geometry);
    generator.generate(mesh);
  }
  else
  {
    dolfin_error("CSGMeshGenerator.cpp",
                 "create boundary mesh from CSG geometry",
                 "Unhandled geometry dimension %d", geometry.dim());
  }
}
//-----------------------------------------------------------------------------
#else
void CSGMeshGenerator::generate(Mesh& mesh,
                                const CSGGeometry& geometry,
                                std::size_t resolution)
{
  dolfin_error("CSGMeshGenerator.cpp",
	       "create mesh from CSG geometry",
	       "Mesh generation not available. Dolfin has been compiled without CGAL.");
}
//-----------------------------------------------------------------------------
void CSGMeshGenerator::generate(BoundaryMesh& mesh,
                                const CSGGeometry& geometry)
{
  dolfin_error("CSGMeshGenerator.cpp",
	       "create boundary mesh from CSG geometry",
	       "Mesh generation not available. Dolfin has been compiled without CGAL.");
}
#endif
