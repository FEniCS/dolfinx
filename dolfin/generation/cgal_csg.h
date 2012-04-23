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
//
// First added:  2012-04-19
// Last changed: 2012-04-19

#ifndef __CGAL_CSG_H
#define __CGAL_CSG_H

#ifdef HAS_CGAL

#include <CGAL/basic.h>
#include <CGAL/Gmpq.h>
//#include <CGAL/Gmpz.h>

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>

#include <CGAL/Polyhedron_3.h>
//#include <CGAL/Polyhedral_mesh_domain_3.h>
#include <CGAL/Polyhedral_mesh_domain_with_features_3.h>
#include <CGAL/make_mesh_3.h>




namespace dolfin
{
  namespace csg
  {
    typedef CGAL::Exact_predicates_exact_constructions_kernel Nef_Kernel;
    typedef CGAL::Exact_predicates_inexact_constructions_kernel Meshing_Kernel;

    // CSG
    //typedef CGAL::Cartesian<CGAL::Gmpq> Nef_Kernel;
    typedef CGAL::Nef_polyhedron_3<Nef_Kernel> Nef_polyhedron_3;
    typedef Nef_polyhedron_3::Point_3 Point_3;
    typedef Nef_polyhedron_3::Plane_3 Plane_3;

    //Meshing

    typedef CGAL::Polyhedron_3<Meshing_Kernel> Polyhedron_3;
    //typedef CGAL::Polyhedral_mesh_domain_with_features_3<Meshing_Kernel> Mesh_domain_3;
    typedef CGAL::Polyhedral_mesh_domain_3<Polyhedron_3, Meshing_Kernel> Mesh_domain_3;

    // Triangulation
    typedef CGAL::Mesh_triangulation_3<Mesh_domain_3>::type Tr;
    typedef CGAL::Mesh_complex_3_in_triangulation_3<Tr> C3t3;
    typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria_3;
  }
}

#endif

#endif
