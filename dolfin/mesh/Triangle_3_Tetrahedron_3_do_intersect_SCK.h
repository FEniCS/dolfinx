// =====================================================================================
//
// Copyright (C) 2010 Andre Massing
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Andre Massing, 2010
//
// First added:  2010-02-09
// Last changed: 2010-04-06
// 
//Author:  Andre Massing (am), massing@simula.no
//Company:  Simula Research Laboratory, Fornebu, Norway
//
// =====================================================================================

#ifndef CGAL_TRIANGLE_3_TETRAHEDRON_3_DO_INTERSECT_SCK_H
#define CGAL_TRIANGLE_3_TETRAHEDRON_3_DO_INTERSECT_SCK_H

#include <CGAL/enum.h>
#include <CGAL/Simple_cartesian.h> 

CGAL_BEGIN_NAMESPACE

#if CGAL_VERSION_NR < 1030601000
namespace CGALi {
#else
namespace internal {
#endif

///Declaration of function template specialization.
template <>
Simple_cartesian<double>::Boolean
do_intersect<Simple_cartesian<double> >(const Simple_cartesian<double>::Triangle_3 &tr,
					const Simple_cartesian<double>::Tetrahedron_3 &tet,
					const Simple_cartesian<double> & k);

///Declaration of function template specialization.
template <>
Simple_cartesian<double>::Boolean
do_intersect<Simple_cartesian<double> >(const Simple_cartesian<double>::Tetrahedron_3 &tet,
					const Simple_cartesian<double>::Triangle_3 &tr,
					const Simple_cartesian<double> & k);

}  //namespace CGALi

CGAL_END_NAMESPACE

#endif // CGAL_TRIANGLE_3_TETRAHEDRON_3_DO_INTERSECT_SCK_H
