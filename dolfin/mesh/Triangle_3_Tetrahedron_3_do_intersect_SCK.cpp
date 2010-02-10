// =====================================================================================
//
// Copyright (C) 2010-02-10  André Massing
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by André Massing, 2010
//
// First added:  2010-02-10
// Last changed: 2010-02-10
// 
//Author:  André Massing (am), massing@simula.no
//Company:  Simula Research Laboratory, Fornebu, Norway
//
// =====================================================================================

#include "Triangle_3_Tetrahedron_3_do_intersect_SCK.h" 

namespace CGAL {

namespace CGALi {

template <>
Simple_cartesian<double>::Boolean
do_intersect<Simple_cartesian<double> >(const Simple_cartesian<double>::Triangle_3 &tr,
             const Simple_cartesian<double>::Tetrahedron_3 &tet,
             const Simple_cartesian<double> & k)
{ 
    typedef Simple_cartesian<double>::Triangle_3 Triangle;
    typedef Simple_cartesian<double>::Point_3    Point;
    
    CGAL_kernel_precondition( ! k.is_degenerate_3_object() (tr) );
    CGAL_kernel_precondition( ! k.is_degenerate_3_object() (tet) );

    if (!tet.has_on_unbounded_side(tr[0])) return true;
    if (!tet.has_on_unbounded_side(tr[1])) return true;
    if (!tet.has_on_unbounded_side(tr[2])) return true;

    if (do_intersect(tr, Triangle(tet[0], tet[1], tet[2]), k)) return true;
    if (do_intersect(tr, Triangle(tet[0], tet[1], tet[3]), k)) return true;
    if (do_intersect(tr, Triangle(tet[0], tet[2], tet[3]), k)) return true;
    if (do_intersect(tr, Triangle(tet[1], tet[2], tet[3]), k)) return true;

    return false;
}
//-----------------------------------------------------------------------------
template <>
Simple_cartesian<double>::Boolean
do_intersect<Simple_cartesian<double> >(const Simple_cartesian<double>::Tetrahedron_3 &tet,
					const Simple_cartesian<double>::Triangle_3 &tr,
					const Simple_cartesian<double> & k)
{
  return do_intersect(tr, tet, k);
}

} // end CGALi

}  //end CGAL
