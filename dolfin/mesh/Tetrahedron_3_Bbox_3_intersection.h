// Copyright (C) 2009 Andre Massing
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
// First added:  2009-09-11
// Last changed: 2010-04-06

#ifndef  TETRAHEDRON_3_BBOX_3_INTERSECTION_INC
#define  TETRAHEDRON_3_BBOX_3_INTERSECTION_INC

#include <CGAL/Bbox_3.h>
#include <CGAL/Tetrahedron_3.h>
//#include <CGAL/AABB_intersections/Bbox_3_triangle_3_do_intersect.h>

CGAL_BEGIN_NAMESPACE

#if CGAL_VERSION_NR < 1030601000
namespace CGALi {
#else
namespace internal {
#endif

  //This code is not optimized!!
  template <class K>
  inline
  bool do_intersect(const typename K::Tetrahedron_3& tet,
		    const CGAL::Bbox_3& bbox,
		    const K & k)
  {
    typedef typename K::Point_3    Point;
    typedef typename K::Triangle_3 Triangle;

    //Check first whether on point of one primitive intersect the other
    if (do_intersect(tet[0],bbox,k)) return true;
    if (!k.has_on_unbounded_side_3_object()(tet,Point(bbox.xmin(),bbox.ymin(),bbox.zmin()))) return true;
//    if (!tet.has_on_unbounded_side(Point(bbox.xmin(),bbox.ymin(),bbox.zmin()))) return true;

    //Otherwise one tetrahedron face must intersect the bbox in order to intersect.
    if (do_intersect(bbox, Triangle(tet[0], tet[1], tet[2]))) return true;
    if (do_intersect(bbox, Triangle(tet[0], tet[1], tet[3]))) return true;
    if (do_intersect(bbox, Triangle(tet[0], tet[2], tet[3]))) return true;
    if (do_intersect(bbox, Triangle(tet[1], tet[2], tet[3]))) return true;

    return false;
  }

  template <class K>
  inline
  bool do_intersect(const CGAL::Bbox_3& bbox,
		    const typename K::Tetrahedron_3& tet,
		    const K & k)
  {
    return  do_intersect(tet, bbox, k);
  }

//template <class K>
//inline
//Object
//intersection(const typename K::Tetrahedron_3 &tet, 
//             const typename K::Bbox_3 &bbox, 
//             const K&)
//{
//    if (do_intersect(tet,bbox)) {
//      return Object();
//    }
//    return Object();
//}

//template <class K>
//inline
//Object
//intersection( const typename K::Bbox_3 &bbox, 
//              const typename K::Tetrahedron_3 &tet, 
//              const K&)
//{
//    if (do_intersect(tet,bbox)) {
//      return Object();
//    }
//    return Object();
//}


} //namespace CGALi

template <class K>
inline
bool do_intersect(const CGAL::Tetrahedron_3<K>& point,
		  const CGAL::Bbox_3& bbox)
{
  return typename K::Do_intersect_3()(point, bbox);
}

template <class K>
inline
bool do_intersect(const CGAL::Bbox_3& bbox,
		  const CGAL::Tetrahedron_3<K>& point)
{
  return typename K::Do_intersect_3()(point, bbox);
}

//template <class K>
//inline Object
//intersection(const Bbox_3 & bbox, const Tetrahedron_3<K> & tet)
//{
//  typedef typename K::Intersect_3 Intersect;
//  return Intersect()(tet, bbox);
//}

//template <class K>
//inline Object
//intersection(const Tetrahedron_3<K> & tet, const Bbox_3 & bbox)
//{
//  typedef typename K::Intersect_3 Intersect;
//  return Intersect()(tet, bbox);
//}

CGAL_END_NAMESPACE


#endif   /* ----- #ifndef TETRAHEDRON_3_BBOX_3_INTERSECTION_INC  ----- */
