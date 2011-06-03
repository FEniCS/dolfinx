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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Andre Massing, 2010
//
// First added:  2010-02-10
// Last changed: 2010-04-06
// 
//Author:  Andre Massing (am), massing@simula.no
//Company:  Simula Research Laboratory, Fornebu, Norway
//
// =====================================================================================

#ifdef HAS_CGAL

#include "Triangle_3_Tetrahedron_3_do_intersect_SCK.h" 

namespace CGAL {

#if CGAL_VERSION_NR < 1030601000
namespace CGALi {
#else
namespace internal {
#endif

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

#endif

