#ifdef HAS_CGAL

#include <CGAL/basic.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_cell_base_with_info_3.h>

#include <CGAL/IO/Polyhedron_iostream.h>

#include <CGAL/Mesh_criteria_3.h>

#include <CGAL/Polyhedral_mesh_domain_with_features_3.h>
#include <CGAL/make_mesh_3.h>

#include <CGAL/version.h>

namespace dolfin
{
  namespace csg
  {
    // Domain 
    typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
    typedef CGAL::Mesh_polyhedron_3<K>::type Polyhedron_3;
    typedef K::Point_3 Point_3;
    typedef K::Vector_3 Vector_3;
    typedef K::Triangle_3 Triangle_3;
    typedef CGAL::Polyhedral_mesh_domain_with_features_3<K> Mesh_domain;

    // Triangulation
    // typedef CGAL::Mesh_triangulation_3<Mesh_domain>::type Tr;
    // typedef CGAL::Mesh_complex_3_in_triangulation_3<
    //   Tr,Mesh_domain::Corner_index,Mesh_domain::Curve_segment_index> C3t3;

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
    typedef CGAL::Regular_triangulation_3<Geom_traits, Tds_mesh> Tr;

    // CGAL 3D mesh typedef
    typedef CGAL::Mesh_complex_3_in_triangulation_3<
    Tr, Mesh_domain::Corner_index, Mesh_domain::Curve_segment_index> C3t3;


    // Criteria
    typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;
  }
}
#endif
