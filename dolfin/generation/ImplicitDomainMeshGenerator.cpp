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
#include <boost/bind.hpp>
#include <boost/function.hpp>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_cell_base_with_info_3.h>
#include <CGAL/make_mesh_3.h>

#include <CGAL/Implicit_surface_3.h>
#include <CGAL/Implicit_mesh_domain_3.h>
#include <CGAL/Mesh_domain_with_polyline_features_3.h>

#include <dolfin/common/MPI.h>
#include <dolfin/geometry/ImplicitSurface.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include "CGALMeshBuilder.h"

#include "ImplicitDomainMeshGenerator.h"

// CGAL kernel typedefs
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;

// Point
typedef K::Point_3 Point_3;

typedef CGAL::Robust_weighted_circumcenter_filtered_traits_3<K>
Geom_traits;

// Call-back function
typedef boost::function<double (Point_3)> Function;

// Mesh domain
typedef CGAL::Implicit_mesh_domain_3<const Function, K> Mesh_domain0;
typedef CGAL::Mesh_domain_with_polyline_features_3<Mesh_domain0> Mesh_domain;

// CGAL 3D triangulation vertex typedefs
typedef CGAL::Triangulation_vertex_base_3<Geom_traits> Tvb3_base;
typedef CGAL::Triangulation_vertex_base_with_info_3<int, Geom_traits,
                                             Tvb3_base> Tvb3;
typedef CGAL::Mesh_vertex_base_3<Geom_traits, Mesh_domain,
                                 Tvb3> Vertex_base;

// CGAL 3D triangulation cell typedefs
typedef CGAL::Triangulation_cell_base_3<Geom_traits> Tcb3_base;
typedef CGAL::Triangulation_cell_base_with_info_3<int, Geom_traits,
                                            Tcb3_base> Tcb3;
typedef CGAL::Mesh_cell_base_3<Geom_traits, Mesh_domain,
                               Tcb3> Cell_base;

// CGAL 3D triangulation typedefs
typedef CGAL::Triangulation_data_structure_3<Vertex_base, Cell_base>
Tds_mesh;
typedef CGAL::Regular_triangulation_3<Geom_traits, Tds_mesh> Tr;

// Triangulation
typedef CGAL::Mesh_complex_3_in_triangulation_3<Tr> C3t3;

// Criteria
typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;


using namespace dolfin;

// Lightweight class for wrapping callback to
// ImplicitSurface::operator()
class ImplicitSurfaceWrapper
{
public:
  ImplicitSurfaceWrapper(const ImplicitSurface& surface) : _surface(surface) {}
  ~ImplicitSurfaceWrapper() {}

  double f(const Point_3& p)
  {
    _p[0] = p[0]; _p[1] = p[1]; _p[2] = p[2];
    return _surface(_p);
  }

private:
  const ImplicitSurface& _surface;
  Point _p;

};

//-----------------------------------------------------------------------------
void ImplicitDomainMeshGenerator::generate(Mesh& mesh,
                                           const ImplicitSurface& surface,
                                           double cell_size)
{
  if (MPI::process_number() == 0)
  {
    // Mesh criteria
    Mesh_criteria criteria(CGAL::parameters::edge_size =cell_size,
                           CGAL::parameters::facet_angle = 30,
                           CGAL::parameters::facet_size =cell_size,
                           CGAL::parameters::cell_size = cell_size);

    // Build CGAL mesh
    C3t3 c3t3 = build_cgal_triangulation<C3t3>(surface, criteria);

    // Build DOLFIN mesh from CGAL 3D mesh/triangulation
    CGALMeshBuilder::build_from_mesh(mesh, c3t3);
  }

  // Build distributed mesh
  MeshPartitioning::build_distributed_mesh(mesh);
}
//-----------------------------------------------------------------------------
void
ImplicitDomainMeshGenerator::generate_surface(Mesh& mesh,
                                              const ImplicitSurface& surface,
                                              double cell_size)

{
  if (MPI::process_number() == 0)
  {
    // CGAL mesh paramters
    Mesh_criteria criteria(CGAL::parameters::edge_size = cell_size,
                           CGAL::parameters::facet_angle = 30,
                           CGAL::parameters::facet_size = cell_size,
                           CGAL::parameters::cell_size = 0.0);

    // Build CGAL mesh
    C3t3 c3t3 = build_cgal_triangulation<C3t3>(surface, criteria);

    // Build surface DOLFIN mesh from CGAL 3D mesh/triangulation
    CGALMeshBuilder::build_surface_mesh_c3t3(mesh, c3t3, &surface);
  }

  // Build distributed mesh
  MeshPartitioning::build_distributed_mesh(mesh);
}
//-----------------------------------------------------------------------------
template<typename X, typename Y>
X ImplicitDomainMeshGenerator::build_cgal_triangulation(const ImplicitSurface& surface,
                                                        const Y& mesh_criteria)
{
  // Wrap implicit surface
  ImplicitSurfaceWrapper surface_wrapper(surface);
  boost::function<double (Point_3)> f(boost::bind(&ImplicitSurfaceWrapper::f,
                                                  &surface_wrapper, _1));

  // Create CGAL bounding sphere
  const Point c = surface.sphere.c;
  dolfin_assert(surface.sphere.r > 0.0);
  K::Sphere_3 bounding_sphere(Point_3(c[0], c[1], c[2]),
                              surface.sphere.r*surface.sphere.r);

  // Create mesh domain
  Mesh_domain domain(f, bounding_sphere, 1.0e-5);

  // Add polylines (if any)
  std::list<std::vector<Point_3> > cgal_polylines;
  std::list<std::vector<Point> >::const_iterator polyline;
  const std::list<std::vector<Point> >& polylines = surface.polylines;
  for (polyline = polylines.begin(); polyline != polylines.end(); ++polyline)
  {
    std::vector<Point_3> _line;
    _line.reserve(polyline->size());
    std::vector<Point>::const_iterator p;
    for (p = polyline->begin(); p != polyline->end(); ++p)
      _line.push_back(Point_3(p->x(), p->y(), p->z()));

    cgal_polylines.push_back(_line);
  }

  // Add polyline features
  domain.add_features(cgal_polylines.begin(), cgal_polylines.end());

  // Generate CGAL mesh
  return CGAL::make_mesh_3<X>(domain, mesh_criteria);
}
//-----------------------------------------------------------------------------

#endif
