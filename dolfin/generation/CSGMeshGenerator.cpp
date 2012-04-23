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

#include "cgal_csg.h"

#include <dolfin/log/log.h>
#include "CSGMeshGenerator.h"
#include "CSGGeometry.h"
#include "CGALMeshBuilder.h"
using namespace dolfin;
#ifdef HAS_CGAL

// Taken from demo/Polyhedron/Scene_nef_polyhedron_item.cpp in the
// CGAL source tree.
// Quick hacks to convert polyhedra from exact to inexact and
// vice-versa
template <class Polyhedron_input, class Polyhedron_output>
struct Copy_polyhedron_to 
  : public CGAL::Modifier_base<typename Polyhedron_output::HalfedgeDS>
{
  Copy_polyhedron_to(const Polyhedron_input& in_poly) 
    : in_poly(in_poly) {}

  void operator()(typename Polyhedron_output::HalfedgeDS& out_hds)
  {
    typedef typename Polyhedron_output::HalfedgeDS Output_HDS;
    typedef typename Polyhedron_input::HalfedgeDS Input_HDS;

    CGAL::Polyhedron_incremental_builder_3<Output_HDS> builder(out_hds);

    typedef typename Polyhedron_input::Vertex_const_iterator Vertex_const_iterator;
    typedef typename Polyhedron_input::Facet_const_iterator  Facet_const_iterator;
    typedef typename Polyhedron_input::Halfedge_around_facet_const_circulator HFCC;

    builder.begin_surface(in_poly.size_of_vertices(),
      in_poly.size_of_facets(),
      in_poly.size_of_halfedges());

    for(Vertex_const_iterator
      vi = in_poly.vertices_begin(), end = in_poly.vertices_end();
      vi != end ; ++vi) 
    {
      typename Polyhedron_output::Point_3 p(::CGAL::to_double( vi->point().x()),
	::CGAL::to_double( vi->point().y()),
	::CGAL::to_double( vi->point().z()));
      builder.add_vertex(p);
    }

    typedef CGAL::Inverse_index<Vertex_const_iterator> Index;
    Index index( in_poly.vertices_begin(), in_poly.vertices_end());

    for(Facet_const_iterator 
      fi = in_poly.facets_begin(), end = in_poly.facets_end();
      fi != end; ++fi) 
    {
      HFCC hc = fi->facet_begin();
      HFCC hc_end = hc;
      //     std::size_t n = circulator_size( hc);
      //     CGAL_assertion( n >= 3);
      builder.begin_facet ();
      do {
	builder.add_vertex_to_facet(index[hc->vertex()]);
	++hc;
      } while( hc != hc_end);
      builder.end_facet();
    }
    builder.end_surface();
  } // end operator()(..)
private:
  const Polyhedron_input& in_poly;
}; // end Copy_polyhedron_to<>
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
  csg::Nef_polyhedron_3 cgal_geometry = geometry.get_cgal_type_3D();
  
  csg::Polyhedron_3 p;
  //cgal_geometry.convert_to_polyhedron(p);

  // Copy_polyhedron_to<csg::Nef_polyhedron_3, csg::Polyhedron_3> converter(cgal_geometry);
  // converter(p);


  // // Create domain
  csg::Mesh_domain_3 domain(p);

  csg::Mesh_criteria_3 criteria(CGAL::parameters::facet_angle=25, 
  				CGAL::parameters::facet_size=0.15,
  				CGAL::parameters::facet_distance=0.008,
  				CGAL::parameters::cell_radius_edge_ratio=3);

  // // Generate CGAL mesh
  csg::C3t3 c3t3 = CGAL::make_mesh_3<csg::C3t3>(domain, criteria);

  // Build DOLFIN mesh from CGAL mesh/triangulation
  //CGALMeshBuilder::build_from_mesh(mesh, c3t3);

  // // Clear mesh
  // mesh.clear();

  // // CGAL triangulation
  // typename T::Triangulation t = cgal_mesh.triangulation();

  // // Get various dimensions
  // const uint gdim = t.finite_vertices_begin()->point().dimension();
  // const uint tdim = t.dimension();
  // const uint num_vertices = t.number_of_vertices();
  // const uint num_cells = cgal_mesh.number_of_cells();

  // // Create a MeshEditor and open
  // dolfin::MeshEditor mesh_editor;
  // mesh_editor.open(mesh, tdim, gdim);
  // mesh_editor.init_vertices(num_vertices);
  // mesh_editor.init_cells(num_cells);

  // // Add vertices to mesh
  // unsigned int vertex_index = 0;
  // typename T::Triangulation::Finite_vertices_iterator v;
  // for (v = t.finite_vertices_begin(); v != t.finite_vertices_end(); ++v)
  // {
  //   // Get vertex coordinates and add vertex to the mesh
  //   Point p;
  //   p[0] = v->point()[0];
  //   p[1] = v->point()[1];
  //   p[2] = v->point()[2];
  
  //   // Add mesh vertex
  //   mesh_editor.add_vertex(vertex_index, p);
  
  //   // Attach index to vertex and increment
  //   v->info() = vertex_index++;
  // }

  // // Sanity check on number of vertices
  // dolfin_assert(vertex_index == num_vertices);
  
  // // Iterate over all cell in triangulation
  // unsigned int cell_index = 0;
  // typename T::Triangulation::Finite_cells_iterator c;
  // for (c = t.finite_cells_begin(); c != t.finite_cells_end(); ++c)
  // {
  //   // Add cell if in CGAL mesh, and increment index
  //   if (cgal_mesh.is_in_complex(c))
  //   {
  //     mesh_editor.add_cell(cell_index++, c->vertex(0)->info(),
  //                                        c->vertex(1)->info(),
  //                                        c->vertex(2)->info(),
  //                                        c->vertex(3)->info());
  //   }
  // }

  // // Sanity check on number of cells
  // dolfin_assert(cell_index == num_cells);

  // // Close mesh editor
  // mesh_editor.close();

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
