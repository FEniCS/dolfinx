#include "CSGCGALMeshGenerator2D.h"

using namespace dolfin;


// typedef CGAL::Polygon_2<csg::Inexact_Kernel> Polygon_2;
// typedef csg::Inexact_Kernel::Point_2 Point_2;


//-----------------------------------------------------------------------------
CSGCGALMeshGenerator2D::CSGCGALMeshGenerator2D(const CSGGeometry& geometry)
  : geometry(geometry)
{}
//-----------------------------------------------------------------------------
CSGCGALMeshGenerator2D::~CSGCGALMeshGenerator2D() {}
//-----------------------------------------------------------------------------

#include "UnitSquare.h"
void CSGCGALMeshGenerator2D::generate(Mesh& mesh) 
{
  // return soemthing
  UnitSquare square(10, 10);
  mesh = square;

  // csg::Nef_polyhedron_2 cgal_geometry = geometry.get_cgal_type_2D();

  // // Create empty CGAL triangulation
  // csg::CDT cdt;

  // // Explore the Nef polyhedron and insert constraints in the triangulation
  // csg::Explorer explorer = cgal_geometry.explorer();
  // csg::Face_const_iterator fit = explorer.faces_begin();
  // for (; fit != explorer.faces_end(); fit++)
  // {
  //   // Skip face if it is not part of polygon
  //   if (! explorer.mark(fit))
  //     continue;

  //   Polygon_2 polygon;
  //   csg::Halfedge_around_face_const_circulator hafc = explorer.face_cycle(fit), done(hafc);
  //   do {
  //     csg::Vertex_const_handle vh = explorer.target(hafc);
  //     polygon.push_back(Point_2(to_double(explorer.point(vh).x()),
  //                               to_double(explorer.point(vh).y())));
  //     hafc++;
  //   } while(hafc != done);
  //   insert_polygon(cdt, polygon);

  //   // FIXME: Holes must be marked as not part of the mesh domain
  //   csg::Hole_const_iterator hit = explorer.holes_begin(fit);
  //   for (; hit != explorer.holes_end(fit); hit++)
  //   {
  //     Polygon_2 hole;
  //     csg::Halfedge_around_face_const_circulator hafc(hit), done(hit);
  //     do {
  //       csg::Vertex_const_handle vh = explorer.target(hafc);
  //       hole.push_back(Point_2(to_double(explorer.point(vh).x()),
  //                              to_double(explorer.point(vh).y())));
  //       hafc++;
  //     } while(hafc != done);
  //     insert_polygon(cdt, hole);
  //   }
  // }

  // // Create mesher
  // csg::CGAL_Mesher_2 mesher(cdt);

  // csg::Mesh_criteria_2 criteria(0.125, 0.25);

  // // Refine CGAL mesh/triangulation
  // mesher.set_criteria(criteria);
  // mesher.refine_mesh();

  // dolfin_assert(cdt.is_valid());

  // // Build DOLFIN mesh from CGAL triangulation
  // CGALMeshBuilder::build(mesh, cdt);
}

//-----------------------------------------------------------------------------
// void insert_polygon(csg::CDT& cdt, const Polygon_2& polygon)
// {
//   if (polygon.is_empty())
//     return;

//   csg::CDT::Vertex_handle v_prev = cdt.insert(*CGAL::cpp0x::prev(polygon.vertices_end()));
//   for (Polygon_2::Vertex_iterator vit = polygon.vertices_begin();
//        vit != polygon.vertices_end(); ++vit)
//   {
//     csg::CDT::Vertex_handle vh = cdt.insert(*vit);
//     cdt.insert_constraint(vh,v_prev);
//     v_prev = vh;
//   }
// }
