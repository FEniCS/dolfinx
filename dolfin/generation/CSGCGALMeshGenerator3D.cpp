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
// First added:  2012-05-10
// Last changed: 2012-05-10


#include "CSGCGALMeshGenerator3D.h"
#include "GeometryToCGALConverter.h"
#include "CGALMeshBuilder.h"
#include <dolfin/log/LogStream.h>
#include "cgal_csg3d.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
CSGCGALMeshGenerator3D::CSGCGALMeshGenerator3D(const CSGGeometry& geometry)
  : geometry(geometry)
{
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
CSGCGALMeshGenerator3D::~CSGCGALMeshGenerator3D() {}
//-----------------------------------------------------------------------------
void CSGCGALMeshGenerator3D::generate(Mesh& mesh) 
{
  csg::Polyhedron_3 p;

  cout << "Converting geometry to cgal types." << endl;
  GeometryToCGALConverter::convert(geometry, p);

  dolfin_assert(p.is_pure_triangle());

  csg::Mesh_domain domain(p);

  cout << "Detecting sharp features" << endl;
  domain.detect_features();

  // Mesh criteria
  csg::Mesh_criteria criteria(CGAL::parameters::edge_size = parameters["edge_size"],
			      CGAL::parameters::facet_angle = parameters["facet_angle"], 
			      CGAL::parameters::facet_size = parameters["facet_size"], 
			      CGAL::parameters::facet_distance = parameters["facet_distance"],
			      CGAL::parameters::cell_radius_edge_ratio = parameters["cell_radius_edge_ratio"], 
			      CGAL::parameters::cell_size = parameters["cell_size"]);
  
  // Mesh generation
  cout << "Generating mesh" << endl;
  csg::C3t3 c3t3 = CGAL::make_mesh_3<csg::C3t3>(domain, criteria);

  // Build DOLFIN mesh from CGAL mesh/triangulation
  CGALMeshBuilder::build_from_mesh(mesh, c3t3);
}
