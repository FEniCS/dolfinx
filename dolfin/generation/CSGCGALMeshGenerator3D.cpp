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
// Modified by Joachim B Haga 2012
//
// First added:  2012-05-10
// Last changed: 2012-10-30


#include "CSGCGALMeshGenerator3D.h"
#include "CSGGeometry.h"
#include "GeometryToCGALConverter.h"
#include "SurfaceFileReader.h"
#include <dolfin/log/LogStream.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/MeshEditor.h>
#include "cgal_csg3d.h"
#include <dolfin/generation/triangulate_polyhedron.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
static void build_dolfin_mesh(const csg::C3t3& c3t3, Mesh& mesh)
{
  typedef csg::C3t3 C3T3;
  typedef typename C3T3::Triangulation Triangulation;
  typedef typename Triangulation::Vertex_handle Vertex_handle;

  // CGAL triangulation
  const Triangulation& triangulation = c3t3.triangulation();

  // Clear mesh
  mesh.clear();

  // Count cells in complex
  uint num_cells = 0;
  for(typename csg::C3t3::Cells_in_complex_iterator cit = c3t3.cells_in_complex_begin();
      cit != c3t3.cells_in_complex_end();
      ++cit) 
  {
    num_cells++;
  }

  // Create and initialize mesh editor
  dolfin::MeshEditor mesh_editor;
  mesh_editor.open(mesh, 3, 3);
  mesh_editor.init_vertices(triangulation.number_of_vertices());
  mesh_editor.init_cells(num_cells);

  // Add vertices to mesh
  dolfin::uint vertex_index = 0;
  std::map<Vertex_handle, dolfin::uint> vertex_id_map;

  for (typename Triangulation::Finite_vertices_iterator 
         cgal_vertex = triangulation.finite_vertices_begin();
       cgal_vertex != triangulation.finite_vertices_end(); ++cgal_vertex)
  {
    vertex_id_map[cgal_vertex] = vertex_index;
    
      // Get vertex coordinates and add vertex to the mesh
    Point p(cgal_vertex->point()[0], cgal_vertex->point()[1], cgal_vertex->point()[2]);
    mesh_editor.add_vertex(vertex_index, p);

    ++vertex_index;
  }

  // Add cells to mesh 
  dolfin::uint cell_index = 0;
  for(typename csg::C3t3::Cells_in_complex_iterator cit = c3t3.cells_in_complex_begin();
      cit != c3t3.cells_in_complex_end();
      ++cit) 
  {
    mesh_editor.add_cell(cell_index,
                         vertex_id_map[cit->vertex(0)],
                         vertex_id_map[cit->vertex(1)],
                         vertex_id_map[cit->vertex(2)],
                         vertex_id_map[cit->vertex(3)]);

    ++cell_index;
  }

  // Close mesh editor
  mesh_editor.close();
}
//-----------------------------------------------------------------------------
CSGCGALMeshGenerator3D::CSGCGALMeshGenerator3D(const CSGGeometry& geometry)
{
  boost::shared_ptr<const CSGGeometry> tmp = reference_to_no_delete_pointer<const CSGGeometry>(geometry);
  this->geometry = tmp;
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
CSGCGALMeshGenerator3D::CSGCGALMeshGenerator3D(boost::shared_ptr<const CSGGeometry> geometry)
  : geometry(geometry)
{
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
CSGCGALMeshGenerator3D::~CSGCGALMeshGenerator3D() {}
//-----------------------------------------------------------------------------
void CSGCGALMeshGenerator3D::generate(Mesh& mesh) const
{
  csg::Polyhedron_3 p;

  cout << "Converting geometry to cgal types." << endl;
  GeometryToCGALConverter::convert(*geometry, p, parameters["remove_degenerated"]);

  dolfin_assert(p.is_pure_triangle());

  csg::Mesh_criteria criteria;
  int mesh_resolution = parameters["mesh_resolution"];
  if (mesh_resolution > 0)
  {
    // Try to compute reasonable parameters
    std::cout << "Bounding box of domain: " << csg::SurfaceFileReader::getBoundingBox(p) << std::endl;
    std::cout << "Radius of bounding sphere: " << csg::SurfaceFileReader::getBoundingSphereRadius(p) << std::endl;
    const double cell_size = csg::SurfaceFileReader::getBoundingSphereRadius(p)/mesh_resolution;

    criteria = csg::Mesh_criteria (CGAL::parameters::edge_size = parameters["edge_size"],
                                   CGAL::parameters::facet_angle = parameters["facet_angle"], 
                                   CGAL::parameters::facet_size = parameters["facet_size"], 
                                   CGAL::parameters::facet_distance = parameters["facet_distance"],
                                   CGAL::parameters::cell_radius_edge_ratio = parameters["cell_radius_edge_ratio"], 
                                   CGAL::parameters::cell_size = parameters["cell_size"]);

  }
  else
  {
    // Mesh criteria
    criteria = csg::Mesh_criteria (CGAL::parameters::edge_size = parameters["edge_size"],
                                   CGAL::parameters::facet_angle = parameters["facet_angle"], 
                                   CGAL::parameters::facet_size = parameters["facet_size"], 
                                   CGAL::parameters::facet_distance = parameters["facet_distance"],
                                   CGAL::parameters::cell_radius_edge_ratio = parameters["cell_radius_edge_ratio"], 
                                   CGAL::parameters::cell_size = parameters["cell_size"]);
  }



  csg::Mesh_domain domain(p);

  if (parameters["detect_sharp_features"])
  {
    cout << "Detecting sharp features" << endl;
    domain.detect_features();
  }

  
  // Mesh generation
  cout << "Generating mesh" << endl;
  csg::C3t3 c3t3 = CGAL::make_mesh_3<csg::C3t3>(domain, criteria,
                                                CGAL::parameters::no_perturb(),
                                                CGAL::parameters::no_exude());

  if (parameters["odt_optimize"])
  {
    cout << "Optimizing mesh by odt optimization" << endl;
    odt_optimize_mesh_3(c3t3, domain);
  }

  if (parameters["lloyd_optimize"])
  {
    cout << "Optimizing mesh by lloyd optimization" << endl;
    lloyd_optimize_mesh_3(c3t3, domain);
  }

  if (parameters["perturb_optimize"])
  {
    cout << "Optimizing mesh by perturbation" << endl;
    // TODO: Set time limit
    CGAL::perturb_mesh_3(c3t3, domain);
  }

  if (parameters["exude_optimize"])
  {
    cout << "Optimizing mesh by sliver exudation" << endl;
    exude_mesh_3(c3t3);
  }

  build_dolfin_mesh(c3t3, mesh);
}
//-----------------------------------------------------------------------------
void CSGCGALMeshGenerator3D::save_off(std::string filename) const
{
  csg::Polyhedron_3 p;

  cout << "Converting geometry to cgal types." << endl;
  GeometryToCGALConverter::convert(*geometry, p);

  cout << "Writing to file " << filename << endl;
  std::ofstream outfile(filename.c_str());

  outfile << p;
  outfile.close();
}
