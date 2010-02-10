// Copyright (C) 2009 Andre Massing 
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-09-11
// Last changed: 2010-02-10

#ifndef CGAL_TRIANGLE_3_TETRAHEDRON_3_DO_INTERSECT_SCK_H
#define CGAL_TRIANGLE_3_TETRAHEDRON_3_DO_INTERSECT_SCK_H

//#include <CGAL/Triangle_3_Tetrahedron_3_do_intersect.h>
#include <CGAL/enum.h>
#include <CGAL/Simple_cartesian.h> 

CGAL_BEGIN_NAMESPACE

namespace CGALi {

template <>
Simple_cartesian<double>::Boolean
do_intersect<Simple_cartesian<double> >(const Simple_cartesian<double>::Triangle_3 &tr,
					const Simple_cartesian<double>::Tetrahedron_3 &tet,
					const Simple_cartesian<double> & k);
//{
//    typedef Simple_cartesian<double>::Triangle_3 Triangle;
//    typedef Simple_cartesian<double>::Point_3    Point;

//    CGAL_kernel_precondition( ! k.is_degenerate_3_object() (tr) );
//    CGAL_kernel_precondition( ! k.is_degenerate_3_object() (tet) );

//    if (!tet.has_on_unbounded_side(tr[0])) return true;
//    if (!tet.has_on_unbounded_side(tr[1])) return true;
//    if (!tet.has_on_unbounded_side(tr[2])) return true;
//    
//    if (do_intersect(tr, Triangle(tet[0], tet[1], tet[2]), k)) return true;
//    if (do_intersect(tr, Triangle(tet[0], tet[1], tet[3]), k)) return true;
//    if (do_intersect(tr, Triangle(tet[0], tet[2], tet[3]), k)) return true;
//    if (do_intersect(tr, Triangle(tet[1], tet[2], tet[3]), k)) return true;

//    return false;
//}

template <>
Simple_cartesian<double>::Boolean
do_intersect<Simple_cartesian<double> >(const Simple_cartesian<double>::Tetrahedron_3 &tet,
					const Simple_cartesian<double>::Triangle_3 &tr,
					const Simple_cartesian<double> & k);
//{
//  return do_intersect(tr, tet, k);
//}

}  //namespace CGALi


//template <>
//inline bool do_intersect<Simple_cartesian<double> >(const Tetrahedron_3<Simple_cartesian<double> > &tet,
//                         const Triangle_3<Simple_cartesian<double> > &tr)
//{
//  return Simple_cartesian<double> ::Do_intersect_3()(tr,tet);
//}

//template <>
//inline bool do_intersect<Simple_cartesian<double> >(const Triangle_3<Simple_cartesian<double> > &tr,
//                         const Tetrahedron_3<Simple_cartesian<double> > &tet)
//{
//  return Simple_cartesian<double> ::Do_intersect_3()(tr,tet);
//}

CGAL_END_NAMESPACE

#endif // CGAL_TRIANGLE_3_TETRAHEDRON_3_DO_INTERSECT_SCK_H
