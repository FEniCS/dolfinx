// Copyright (C) 2013 Garth N. Wells
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
// First added:  2013-05-04
// Last changed:

#ifdef HAS_CGAL

#include <vector>
#include <boost/bind.hpp>
#include <boost/function.hpp>

#include <CGAL/Complex_2_in_triangulation_3.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Implicit_surface_3.h>
#include <CGAL/make_surface_mesh.h>
#include <CGAL/Robust_circumcenter_traits_3.h>
#include <CGAL/Surface_mesh_default_criteria_3.h>
#include <CGAL/Surface_mesh_default_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>

#include <dolfin/common/MPI.h>
#include <dolfin/geometry/ImplicitSurface.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/MeshPartitioning.h>

#include "CGALMeshBuilder.h"
#include "SurfaceMeshGenerator.h"

// CGAL kernel typedefs
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;

// CGAL domain typedefs
typedef CGAL::Regular_triangulation_euclidean_traits_3<K> Mesh_domain;
typedef CGAL::Robust_circumcenter_traits_3<K> Geom_traits;

// CGAL 3D triangulation vertex typedefs (so we can attach vertex indices)
typedef CGAL::Triangulation_vertex_base_3<Geom_traits> Tvb3test_base;
typedef CGAL::Triangulation_vertex_base_with_info_3<int, Geom_traits, Tvb3test_base> Tvb3;
typedef CGAL::Complex_2_in_triangulation_vertex_base_3<Geom_traits, Tvb3> Vb_surface;

// CGAL cell type
typedef CGAL::Surface_mesh_cell_base_3<Geom_traits> Cb_surface;
typedef CGAL::Triangulation_cell_base_with_circumcenter_3<Geom_traits, Cb_surface> Cb_with_circumcenter;

// CGAL Triangulation
typedef CGAL::Triangulation_data_structure_3<Vb_surface, Cb_with_circumcenter> Tds_surface;
typedef CGAL::Delaunay_triangulation_3<Geom_traits, Tds_surface> Tr_surface;

// Mesh criteria
typedef CGAL::Surface_mesh_default_criteria_3<Tr_surface> Mesh_criteria;

// c2t3 (a complex)
typedef CGAL::Surface_mesh_complex_2_in_triangulation_3<Tr_surface> C2t3;

typedef Tr_surface::Geom_traits GT;
typedef GT::Sphere_3 Sphere_3;
typedef GT::Point_3 Point_3;
typedef GT::FT FT;

// Call-back function
typedef boost::function<FT (Point_3)> Function;

typedef CGAL::Implicit_surface_3<GT, Function> Surface_3;


using namespace dolfin;

// Lightweight class for wrapping callback to ImplicitSurface::value
class ImplicitSurfaceWrapper
{
public:
  ImplicitSurfaceWrapper(const ImplicitSurface& surface) : _surface(surface) {}
  ~ImplicitSurfaceWrapper() {}

  FT f(const Point_3& p)
  {
    _p[0] = p[0]; _p[1] = p[1]; _p[2] = p[2];
    return _surface(_p);
  }

private:
  Point _p;
  const ImplicitSurface& _surface;
};

//-----------------------------------------------------------------------------
/*
void SurfaceMeshGenerator::generate(Mesh& mesh, const ImplicitSurface& surface,
                                    double min_angle, double max_radius,
                                    double max_distance,
                                    std::size_t num_initial_points)
{
  cout << "Inside  SurfaceMeshGenerator::generat (0)" << endl;

  // Bind implicit surface value member function
  ImplicitSurfaceWrapper surface_wrapper(surface);
  boost::function<FT (Point_3)> f(boost::bind(&ImplicitSurfaceWrapper::f,
                                             &surface_wrapper, _1));

  // Create CGAL bounding sphere
  const Point c = surface.sphere.c;
  Sphere_3 bounding_sphere(Point_3(c[0], c[1], c[2]), surface.sphere.r );

  // Create CGAL implicit surface
  Surface_3 cgal_implicit_surface(f, bounding_sphere);

  // Meshing criteria
  CGAL::Surface_mesh_default_criteria_3<Tr_surface>
    criteria(min_angle, max_radius, max_distance);


  cout << "Inside  SurfaceMeshGenerator::generat (1)" << endl;

  // Build CGAL mesh
  Tr_surface tr;
  C2t3 c2t3(tr);
  if (surface.type == "manifold")
  {
    CGAL::make_surface_mesh(c2t3, cgal_implicit_surface, criteria,
                            CGAL::Manifold_tag(), num_initial_points);
  }
  else if (surface.type == "manifold_with_boundary")
  {
    CGAL::make_surface_mesh(c2t3, cgal_implicit_surface, criteria,
                            CGAL::Manifold_with_boundary_tag(),
                            num_initial_points);

  }
  else if (surface.type == "non_manifold")
  {
    CGAL::make_surface_mesh(c2t3, cgal_implicit_surface, criteria,
                            CGAL::Non_manifold_tag(),
                            num_initial_points);
  }
  else
  {
   dolfin_error("SurfaceMeshGenerator.cpp",
                 "generate surface mesh",
                 "Unknown surface type \"%s\"", surface.type.c_str());
  }

  cout << "Inside  SurfaceMeshGenerator::generat (2)" << endl;

  // Build DOLFIN mesh from CGAL mesh/triangulation
  CGALMeshBuilder::build_mesh_c2t3(mesh, c2t3);
}
*/
//-----------------------------------------------------------------------------
void SurfaceMeshGenerator::generate_surface(Mesh& mesh, const ImplicitSurface& surface,
                                    double min_angle, double max_radius,
                                    double max_distance,
                                    std::size_t num_initial_points)
{
  // Bind implicit surface value member function
  ImplicitSurfaceWrapper surface_wrapper(surface);
  boost::function<FT (Point_3)> f(boost::bind(&ImplicitSurfaceWrapper::f,
                                             &surface_wrapper, _1));

  // Create CGAL bounding sphere
  const Point c = surface.sphere.c;
  Sphere_3 bounding_sphere(Point_3(c[0], c[1], c[2]), surface.sphere.r );

  // Create CGAL implicit surface
  Surface_3 cgal_implicit_surface(f, bounding_sphere);

  // Meshing criteria
  CGAL::Surface_mesh_default_criteria_3<Tr_surface>
    criteria(min_angle, max_radius, max_distance);

  // Build CGAL mesh
  Tr_surface tr;
  C2t3 c2t3(tr);
  if (surface.type == "manifold")
  {
    CGAL::make_surface_mesh(c2t3, cgal_implicit_surface, criteria,
                            CGAL::Manifold_tag(), num_initial_points);
  }
  else if (surface.type == "manifold_with_boundary")
  {
    CGAL::make_surface_mesh(c2t3, cgal_implicit_surface, criteria,
                            CGAL::Manifold_with_boundary_tag(),
                            num_initial_points);

  }
  else if (surface.type == "non_manifold")
  {
    CGAL::make_surface_mesh(c2t3, cgal_implicit_surface, criteria,
                            CGAL::Non_manifold_tag(),
                            num_initial_points);
  }
  else
  {
   dolfin_error("SurfaceMeshGenerator.cpp",
                 "generate surface mesh",
                 "Unknown surface type \"%s\"", surface.type.c_str());
  }

  // Build DOLFIN mesh from CGAL mesh/triangulation
  CGALMeshBuilder::build_surface_mesh_c2t3(mesh, c2t3);
}
//-----------------------------------------------------------------------------
#endif
