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
// Last changed: 2012-02-16

#ifdef HAS_CGAL

#include <vector>
#include <boost/function.hpp>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Mesh_polyhedron_3.h>
#include <CGAL/Polyhedral_mesh_domain_with_features_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_cell_base_with_info_3.h>
#include <CGAL/make_mesh_3.h>

// The below two files are from the CGAL demos. Path can be changed
// once they are included with the CGAL code.
#include "triangulate_polyhedron.h"
#include "compute_normal.h"

#include <dolfin/common/MPI.h>
#include <dolfin/geometry/ImplicitSurface.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include "CGALMeshBuilder.h"
#include "PolyhedralMeshGenerator.h"


// CGAL kernel typedefs
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;

// CGAL domain typedefs
typedef CGAL::Polyhedral_mesh_domain_with_features_3<K> Mesh_domain;

typedef CGAL::Robust_weighted_circumcenter_filtered_traits_3<K> Geom_traits;

// CGAL 3D triangulation vertex typedefs
typedef CGAL::Triangulation_vertex_base_3<Geom_traits> Tvb3test_base;
typedef CGAL::Triangulation_vertex_base_with_info_3<int, Geom_traits, Tvb3test_base> Tvb3test;
typedef CGAL::Mesh_vertex_base_3<Geom_traits, Mesh_domain, Tvb3test> Vertex_base;

// CGAL 3D triangulation cell typedefs
typedef CGAL::Triangulation_cell_base_3<Geom_traits> Tcb3test_base;
typedef CGAL::Triangulation_cell_base_with_info_3<int, Geom_traits, Tcb3test_base> Tcb3test;
typedef CGAL::Mesh_cell_base_3<Geom_traits, Mesh_domain, Tcb3test> Cell_base;

// CGAL 3D triangulation typedefs
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
               const std::vector<std::vector<std::size_t> >& facets)
             : _vertices(vertices), _facets(facets)  {}

  void operator()(HDS& hds)
  {
    // Check that hds is a valid polyhedral surface
    CGAL::Polyhedron_incremental_builder_3<HDS> B(hds, true);

    // Initialise polyhedron building
    B.begin_surface(_vertices.size(), _facets.size(), 0);

    typedef typename HDS::Vertex CVertex;
    typedef typename CVertex::Point CPoint;

    // Add vertices
    std::vector<Point>::const_iterator p;
    for (p = _vertices.begin(); p != _vertices.end(); ++p)
      B.add_vertex(CPoint(p->x(), p->y(), p->z()));

    // Add facets
    std::vector<std::vector<std::size_t> >::const_iterator f;
    for (f = _facets.begin(); f != _facets.end(); ++f)
    {
      // Add facet vertices
      B.begin_facet();
      for (std::size_t i = 0; i < f->size(); ++i)
        B.add_vertex_to_facet((*f)[i]);
      B.end_facet();
    }

    // Finalise
    B.end_surface();
  }

private:

  const std::vector<Point>& _vertices;
  const std::vector<std::vector<std::size_t> >& _facets;

};
//-----------------------------------------------------------------------------
void PolyhedralMeshGenerator::generate(Mesh& mesh, const std::string off_file,
                                       double cell_size,
                                       bool detect_sharp_features)
{
  // Generate CGAL mesh on root process
  if (MPI::process_number() == 0)
  {
    // Create empty CGAL polyhedron
    Polyhedron p;

    // Read polyhedron from file
    std::ifstream p_file(off_file.c_str());
    if (!p_file)
    {
      dolfin_error("PolyhedralMeshGenerator.cpp",
                   "open .off file to read 3D geometry",
                   "Failed to open file");
    }
    p_file >> p;

    // Generate mesh
    cgal_generate(mesh, p, cell_size, detect_sharp_features);
  }

  // Build distributed mesh
  MeshPartitioning::build_distributed_mesh(mesh);
}
//-----------------------------------------------------------------------------
void PolyhedralMeshGenerator::generate(Mesh& mesh,
                        const std::vector<Point>& vertices,
                        const std::vector<std::vector<std::size_t> >& facets,
                        double cell_size, bool detect_sharp_features)
{
  // Generate CGAL mesh on root process
  if (MPI::process_number() == 0)
  {
    // Create empty CGAL polyhedron
    Polyhedron p;

    // Build CGAL polyhedron
    BuildSurface<HalfedgeDS> poly_builder(vertices, facets);
    p.delegate(poly_builder);

    // Generate mesh
    cgal_generate(mesh, p, cell_size, detect_sharp_features);
  }

  // Build distributed mesh
  MeshPartitioning::build_distributed_mesh(mesh);
}
//-----------------------------------------------------------------------------
void PolyhedralMeshGenerator::generate_surface_mesh(Mesh& mesh,
                         const std::vector<Point>& vertices,
                         const std::vector<std::vector<std::size_t> >& facets,
                         double cell_size, bool detect_sharp_features)
{
  // Generate CGAL mesh on root process
  if (MPI::process_number() == 0)
  {
    // Create empty CGAL polyhedron
    Polyhedron p;

    // Build CGAL polyhedron
    BuildSurface<HalfedgeDS> poly_builder(vertices, facets);
    p.delegate(poly_builder);

    // Generate surface mesh
    cgal_generate_surface_mesh(mesh, p, cell_size, detect_sharp_features);
  }

  // Build distributed mesh
  MeshPartitioning::build_distributed_mesh(mesh);
}
//-----------------------------------------------------------------------------
void PolyhedralMeshGenerator::generate_surface_mesh(Mesh& mesh,
                                                    const std::string off_file,
                                                    double cell_size,
                                                    bool detect_sharp_features)
{
  if (MPI::num_processes() == 0)
  {
    // Create empty CGAL polyhedron
    Polyhedron p;

    // Read polyhedron from file
    std::ifstream p_file(off_file.c_str());
    if (!p_file)
    {
      dolfin_error("PolyhedralMeshGenerator.cpp",
                   "open .off file to read 3D geometry",
                   "Failed to open file");
    }
    p_file >> p;

    // Generate mesh
    cgal_generate_surface_mesh(mesh, p, cell_size, detect_sharp_features);
  }

  // Build distributed mesh
  MeshPartitioning::build_distributed_mesh(mesh);
}
//-----------------------------------------------------------------------------
template<typename T>
void PolyhedralMeshGenerator::cgal_generate(Mesh& mesh, T& p,
                                            double cell_size,
                                            bool detect_sharp_features)
{
  // Check if any facets are not triangular and triangulate if
  // necessary.  The CGAL mesh generation only supports polyhedra with
  // triangular surface facets.

  typename Polyhedron::Facet_iterator facet;
  for (facet = p.facets_begin(); facet != p.facets_end(); ++facet)
  {
    // Check if there is a non-triangular facet
    if (!facet->is_triangle())
    {
      CGAL::triangulate_polyhedron(p);
      break;
    }
  }

  // Create domain from polyhedron
  Mesh_domain domain(p);

  // Get sharp features
  if (detect_sharp_features)
    domain.detect_features();

  const Mesh_criteria criteria(CGAL::parameters::facet_angle = 25,
                               CGAL::parameters::facet_size = cell_size,
                               CGAL::parameters::cell_radius_edge_ratio = 3.0,
                               CGAL::parameters::edge_size = cell_size);

  // Generate CGAL mesh
  C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria);

  // Build DOLFIN mesh from CGAL mesh/triangulation
  CGALMeshBuilder::build_from_mesh(mesh, c3t3);
}
//-----------------------------------------------------------------------------
template<typename T>
void PolyhedralMeshGenerator::cgal_generate_surface_mesh(Mesh& mesh, T& p,
                                                         double cell_size,
                                                         bool detect_sharp_features)
{
  // Check if any facets are not triangular and triangulate if
  // necessary.  The CGAL mesh generation only supports polyhedra with
  // triangular surface facets.
  typename Polyhedron::Facet_iterator facet;
  for (facet = p.facets_begin(); facet != p.facets_end(); ++facet)
  {
    // Check if there is a non-triangular facet
    if (!facet->is_triangle())
    {
      CGAL::triangulate_polyhedron(p);
      continue;
    }
  }

  // Create domain from polyhedron
  Mesh_domain domain(p);

  // Get sharp features
  if (detect_sharp_features)
    domain.detect_features();

  // Mesh criteria (produces no interior vertices)
  const Mesh_criteria criteria(CGAL::parameters::facet_angle = 25,
                               CGAL::parameters::facet_size = cell_size,
                               CGAL::parameters::cell_radius_edge = 0,
                               CGAL::parameters::edge_size=0.1,
                               CGAL::parameters::cell_size=0);

  // Generate CGAL mesh
  C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria);

  // Build surface DOLFIN mesh from CGAL 3D mesh/triangulation
  CGALMeshBuilder::build_surface_mesh_c3t3(mesh, c3t3);
}
//-----------------------------------------------------------------------------

#endif
