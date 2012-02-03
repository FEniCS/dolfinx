// Copyright (C) 2012 Garth N. Wells
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
// First added:  2012-02-03
// Last changed:

#ifdef HAS_CGAL

#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Mesh_polyhedron_3.h>
#include <CGAL/Polyhedral_mesh_domain_with_features_3.h>
#include <CGAL/Triangulation_cell_base_with_info_3.h>
#include <CGAL/make_mesh_3.h>

#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/Point.h>
#include "PolyhedralMeshGenerator.h"

#include "CGALMeshBuilder.h"

// CGAL kernel typedefs
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;

// CGAL domain typedefs
typedef CGAL::Polyhedral_mesh_domain_with_features_3<K> Mesh_domain;

typedef CGAL::Robust_weighted_circumcenter_filtered_traits_3<K> Geom_traits;

// CGAL 3D triangulation typedefs
typedef CGAL::Triangulation_vertex_base_3<Geom_traits> Tvb3test_base;
typedef CGAL::Triangulation_vertex_base_with_info_3<int, K, Tvb3test_base> Tvb3test;

typedef CGAL::Triangulation_cell_base_3<Geom_traits> Tcb3test_base;
typedef CGAL::Triangulation_cell_base_with_info_3<int, K, Tcb3test_base> Tcb3test;
typedef CGAL::Mesh_vertex_base_3<Geom_traits, Mesh_domain, Tvb3test> Vertex_base;
typedef CGAL::Mesh_cell_base_3<Geom_traits, Mesh_domain, Tcb3test> Cell_base;
typedef CGAL::Triangulation_data_structure_3<Vertex_base, Cell_base> Tds_mesh;
typedef CGAL::Regular_triangulation_3<Geom_traits, Tds_mesh>             Tr;

// CGAL 3D mesh typedef
typedef CGAL::Mesh_complex_3_in_triangulation_3<
  Tr, Mesh_domain::Corner_index, Mesh_domain::Curve_segment_index> C3t3;

// Mesh criteria
typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;

// Typedefs for building CGAL polyhedron from list of facets
typedef CGAL::Mesh_polyhedron_3<K>::Type Polyhedron;
typedef Polyhedron::Facet_iterator Facet_iterator;
typedef Polyhedron::Halfedge_around_facet_circulator Halfedge_facet_circulator;
typedef Polyhedron::HalfedgeDS HalfedgeDS;

using namespace dolfin;

//-----------------------------------------------------------------------------
template <class HDS>
class BuildSurface : public CGAL::Modifier_base<HDS>
{

  // This class is used to build a CGAL polyhedron from a list of
  // triangular facets. It requires defintion of the member function
  //
  //     void operator()(HDS& hds);

public:

  BuildSurface(const std::vector<Point>& vertices,
               const std::vector<std::vector<unsigned int> >& facets)
             : vertices(vertices), facets(facets)  {}

  void operator()(HDS& hds)
  {
    // Check that hds is a valid polyhedral surface
    CGAL::Polyhedron_incremental_builder_3<HDS> B(hds, true);

    // Initialise polyhedron building
    B.begin_surface(vertices.size(), facets.size(), 0);

    typedef typename HDS::Vertex CVertex;
    typedef typename CVertex::Point CPoint;

    // Add vertices
    std::vector<Point>::const_iterator p;
    for (p = vertices.begin(); p != vertices.end(); ++p)
      B.add_vertex(CPoint(p->x(), p->y(), p->z()));

    // Add facets
    std::vector<std::vector<unsigned int> >::const_iterator f;
    for (f = facets.begin(); f != facets.end(); ++f)
    {
      // Check that facets are triangular
      if (f->size() != 3)
      {
        dolfin_error("PolyhedralMeshGenerator.cpp",
               "build CGAL polyhedron from points and facets",
               "CGAL can only use polyhedra composed of triangular facets for mesh generation)");
      }

      // Add facet vertices
      B.begin_facet();
      for (unsigned int i = 0; i < f->size(); ++i)
        B.add_vertex_to_facet((*f)[i]);
      B.end_facet();
    }

    // Finalise
    B.end_surface();
  }

private:

  const std::vector<Point>& vertices;
  const std::vector<std::vector<unsigned int> >& facets;

};
//-----------------------------------------------------------------------------
void PolyhedralMeshGenerator::generate(Mesh& mesh, const std::string off_file,
                                       double cell_size)
{
  // Create domain from file
  Mesh_domain domain(off_file);

  // Get sharp features
  domain.detect_features();

  // Generate mesh
  cgal_generate(mesh, domain, cell_size);
}
//-----------------------------------------------------------------------------
void PolyhedralMeshGenerator::generate(Mesh& mesh,
                        const std::vector<Point>& vertices,
                        const std::vector<std::vector<unsigned int> >& facets,
                        double cell_size)
{
  // Create empty CGAL polyhedron
  Polyhedron p;

  // Build CGAL polyhedron
  BuildSurface<HalfedgeDS> poly_builder(vertices, facets);
  p.delegate(poly_builder);

  // Create domain from polyhedron
  Mesh_domain domain(p);

  // Get sharp features
  domain.detect_features();

  // Generate mesh
  cgal_generate(mesh, domain, cell_size);
}
//-----------------------------------------------------------------------------
template<typename T>
void PolyhedralMeshGenerator::cgal_generate(Mesh& mesh, const T& domain,
                                            double cell_size)
{
  // Mesh criteria
  /*
  Mesh_criteria criteria(edge_size = 0.125,
                         facet_angle = 25, facet_size = 0.15,
                         facet_distance = 0.05,
                         cell_radius_edge_ratio = 3, cell_size = 0.05);
  */

  /*
  Mesh_criteria criteria(CGAL::parameters::edge_size = 0.0025,
                         CGAL::parameters::facet_angle = 25,
                         CGAL::parameters::facet_size = 0.005,
                         CGAL::parameters::facet_distance = 0.0005,
                         CGAL::parameters::cell_radius_edge_ratio = 3,
                         CGAL::parameters::cell_size = 0.005);
  */

  //const Mesh_criteria criteria(CGAL::parameters::edge_size=cell_size,
  //                             CGAL::parameters::cell_size=cell_size);

  const Mesh_criteria criteria(CGAL::parameters::facet_angle = 25,
                               CGAL::parameters::facet_size = cell_size,
                               CGAL::parameters::cell_radius_edge_ratio = 3,
                               CGAL::parameters::edge_size=cell_size);

  // Generate CGAL mesh
  C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria);

  // Build DOLFIN mesh from CGAL mesh/triangulation
  CGALMeshBuilder::build_from_mesh(mesh, c3t3);
}
//-----------------------------------------------------------------------------

#endif
