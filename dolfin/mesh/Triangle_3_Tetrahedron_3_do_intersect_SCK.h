// =====================================================================================
//
// Copyright (C) 2010-02-09  André Massing
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by André Massing, 2010
//
// First added:  2010-02-09
// Last changed: 2010-02-10
// 
//Author:  André Massing (am), massing@simula.no
//Company:  Simula Research Laboratory, Fornebu, Norway
//
// =====================================================================================

#ifndef CGAL_TRIANGLE_3_TETRAHEDRON_3_DO_INTERSECT_SCK_H
#define CGAL_TRIANGLE_3_TETRAHEDRON_3_DO_INTERSECT_SCK_H

#include <CGAL/enum.h>
#include <CGAL/Simple_cartesian.h> 

CGAL_BEGIN_NAMESPACE

namespace CGALi {

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
