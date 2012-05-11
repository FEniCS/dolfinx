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
{}
//-----------------------------------------------------------------------------
CSGCGALMeshGenerator3D::~CSGCGALMeshGenerator3D() {}
//-----------------------------------------------------------------------------
void CSGCGALMeshGenerator3D::generate(Mesh& mesh) 
{
  #ifdef NDEBUG
  cout << "Debug disabled" << endl;
  #else
  cout << "Debug enabled" << endl;
  #endif

  std::cout << "CGAL version: " << CGAL_VERSION_STR << std::endl;

  csg::Polyhedron_3 p;

  cout << "Converting geometry to cgal types." << endl;
  GeometryToCGALConverter::convert(geometry, p);
  // csg::Point_3 a(0.0, 0.0, 0.0);
  // csg::Point_3 b(1.0, 0.0, 0.0);
  // csg::Point_3 c(1.0, 1.0, 0.0);
  // csg::Point_3 d(0.0, 0.0, 1.0);
  // p.make_tetrahedron(a, b, c, d);

  csg::Mesh_domain domain(p);

  cout << "Detecting sharp features" << endl;
  domain.detect_features();

  // Mesh criteria
  csg::Mesh_criteria criteria(CGAL::parameters::edge_size = 0.025,
			      CGAL::parameters::facet_angle = 25, 
			      CGAL::parameters::facet_size = 0.05, 
			      CGAL::parameters::facet_distance = 0.005,
			      CGAL::parameters::cell_radius_edge_ratio = 3, 
			      CGAL::parameters::cell_size = 0.05);

  cout << "Generating mesh" << endl;
  
  // Mesh generation
  csg::C3t3 c3t3 = CGAL::make_mesh_3<csg::C3t3>(domain, criteria);

  // Output
  std::ofstream medit_file("out.mesh");
  c3t3.output_to_medit(medit_file);

  // // Build DOLFIN mesh from CGAL mesh/triangulation
  CGALMeshBuilder::build_from_mesh(mesh, c3t3);
}
