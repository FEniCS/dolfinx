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
// First added:  2012-02-02
// Last changed:

#ifndef __DOLFIN_CGAL_TRIANGULATE_H
#define __DOLFIN_CGAL_TRIANGULATE_H

#ifdef HAS_CGAL

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_2.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Triangulation_3.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_cell_base_with_info_3.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>

namespace dolfin
{
  /// CGAL typedefs used for triangulation. This file should not be
  /// included globally

  typedef CGAL::Exact_predicates_inexact_constructions_kernel K;

  // 2D triangulation typedefs
  typedef CGAL::Triangulation_vertex_base_2<K> Tvb2base;
  typedef CGAL::Triangulation_vertex_base_with_info_2<unsigned int, K, Tvb2base> Tvb2;
  typedef CGAL::Triangulation_face_base_2<K> Tfb2;
  typedef CGAL::Triangulation_data_structure_2<Tvb2, Tfb2> Tds2;
  typedef CGAL::Delaunay_triangulation_2<K, Tds2> Triangulation2;

  // 3D triangulation typedefs
  typedef CGAL::Triangulation_vertex_base_3<K> Tvb3base;
  typedef CGAL::Triangulation_vertex_base_with_info_3<unsigned int, K, Tvb3base> Tvb3;
  typedef CGAL::Triangulation_cell_base_3<K> Tfb3;
  typedef CGAL::Triangulation_data_structure_3<Tvb3, Tfb3> Tds3;
  typedef CGAL::Delaunay_triangulation_3<K, Tds3> Triangulation3;

  // FIXME: remove duplicated below

  // 2D meshing typedefs
  typedef CGAL::Triangulation_vertex_base_2<K> Vbase;
  typedef CGAL::Triangulation_vertex_base_with_info_2<unsigned int, K, Vbase> Vb;
  typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
  typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
  typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds> CDT;
  typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;
  typedef CGAL::Delaunay_mesher_2<CDT, Criteria> CGAL_Mesher;

  typedef CDT::Vertex_handle Vertex_handle;
  typedef CDT::Point CGAL_Point;

}

#endif
#endif
