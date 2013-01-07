#ifdef HAS_CGAL

#define CGAL_NO_DEPRECATED_CODE
#define CGAL_MESH_3_VERBOSE
//#define PROTECTION_DEBUG

#define CGAL_MESH_3_NO_DEPRECATED_SURFACE_INDEX
#define CGAL_MESH_3_NO_DEPRECATED_C3T3_ITERATORS

#include <CGAL/basic.h>

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Nef_polyhedron_3.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_cell_base_with_info_3.h>

#include <CGAL/IO/Polyhedron_iostream.h>
#include <CGAL/Bbox_3.h>

#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Polyhedral_mesh_domain_with_features_3.h>
#include <CGAL/make_mesh_3.h>

namespace dolfin
{
  namespace csg
  {

    // Exact polyhedron
    typedef CGAL::Exact_predicates_exact_constructions_kernel Exact_Kernel;
    typedef Exact_Kernel::Triangle_3 Exact_Triangle_3;
    typedef CGAL::Nef_polyhedron_3<Exact_Kernel> Nef_polyhedron_3;
    typedef CGAL::Polyhedron_3<Exact_Kernel> Exact_Polyhedron_3;
    typedef Exact_Polyhedron_3::HalfedgeDS Exact_HalfedgeDS;
    typedef Nef_polyhedron_3::Point_3 Exact_Point_3;

    // Domain
    typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
    typedef CGAL::Mesh_polyhedron_3<K>::type Polyhedron_3;
    typedef K::Point_3 Point_3;
    typedef K::Vector_3 Vector_3;
    typedef K::Triangle_3 Triangle_3;
    typedef CGAL::Polyhedral_mesh_domain_with_features_3<K> Mesh_domain;

    // Triangulation
    typedef CGAL::Mesh_triangulation_3<Mesh_domain>::type Tr;
    typedef CGAL::Mesh_complex_3_in_triangulation_3<
      Tr,Mesh_domain::Corner_index,Mesh_domain::Curve_segment_index> C3t3;

    // Criteria
    typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;
  }
}
#endif
