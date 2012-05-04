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
// Modified by Benjamin Kehlet, 2012
// Modified by Johannes Ring, 2012
//
// First added:  2012-04-19
// Last changed: 2012-05-03

#ifndef __CGAL_CSG_H
#define __CGAL_CSG_H

#ifdef HAS_CGAL

#include <CGAL/basic.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_cell_base_with_info_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>
//#include <CGAL/Polyhedral_mesh_domain_3.h>
#include <CGAL/Polyhedral_mesh_domain_with_features_3.h>
#include <CGAL/make_mesh_3.h>

#include <CGAL/Gmpq.h>
#include <CGAL/Lazy_exact_nt.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Bounded_kernel.h>
#include <CGAL/Nef_polyhedron_2.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>

namespace dolfin
{
  namespace csg
  {
    typedef CGAL::Exact_predicates_exact_constructions_kernel Exact_Kernel;
    typedef CGAL::Exact_predicates_inexact_constructions_kernel Inexact_Kernel;

    // CSG
    typedef CGAL::Nef_polyhedron_3<Exact_Kernel> Nef_polyhedron_3;
    typedef CGAL::Polyhedron_3<Exact_Kernel> Exact_Polyhedron_3;
    typedef Exact_Polyhedron_3::HalfedgeDS Exact_HalfedgeDS;
    typedef Nef_polyhedron_3::Point_3 Point_3;
    typedef Nef_polyhedron_3::Plane_3 Plane_3;

    //Meshing
    typedef CGAL::Mesh_polyhedron_3<Inexact_Kernel>::Type Polyhedron_3;
    typedef CGAL::Polyhedral_mesh_domain_with_features_3<Inexact_Kernel> Mesh_domain;
    typedef CGAL::Robust_weighted_circumcenter_filtered_traits_3<Inexact_Kernel> Geom_traits;

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

    // CSG 2D
    typedef CGAL::Lazy_exact_nt<CGAL::Gmpq> FT;
    typedef CGAL::Simple_cartesian<FT> EKernel;
    typedef CGAL::Bounded_kernel<EKernel> Extended_kernel;
    typedef CGAL::Nef_polyhedron_2<Extended_kernel> Nef_polyhedron_2;
    typedef Nef_polyhedron_2::Point Nef_point_2;

    // Explorer typedefs for Nef_polyhedron_2
    typedef Nef_polyhedron_2::Explorer Explorer;
    typedef Explorer::Face_const_iterator Face_const_iterator;
    typedef Explorer::Hole_const_iterator Hole_const_iterator;
    typedef Explorer::Halfedge_around_face_const_circulator Halfedge_around_face_const_circulator;
    typedef Explorer::Vertex_const_handle Vertex_const_handle;

    // CGAL 2D triangulation vertex typedefs
    typedef CGAL::Triangulation_vertex_base_2<Inexact_Kernel> Vb;
    typedef CGAL::Triangulation_vertex_base_with_info_2<unsigned int, Inexact_Kernel, Vb> Vbb;

    // CGAL 2D triangulation face typedefs
    typedef CGAL::Delaunay_mesh_face_base_2<Inexact_Kernel> Fb;

    // CGAL 2D triangulation typedefs
    typedef CGAL::Triangulation_data_structure_2<Vbb, Fb> TDS;
    typedef CGAL::Constrained_Delaunay_triangulation_2<Inexact_Kernel, TDS> CDT;

    // 2D Mesh criteria
    typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Mesh_criteria_2;

    // 2D Meshing
    typedef CGAL::Delaunay_mesher_2<CDT, Mesh_criteria_2> CGAL_Mesher_2;
  }
}

#endif

#endif
